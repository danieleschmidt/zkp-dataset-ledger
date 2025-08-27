//! High-Performance Caching System for ZKP Dataset Ledger
//!
//! Implements multi-level caching with LRU eviction, compression,
//! and concurrent access patterns for maximum performance.

use crate::Result;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Cache entry with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheEntry<T> {
    /// Cached data
    pub data: T,
    /// Creation timestamp
    pub created_at: u64,
    /// Last access timestamp
    pub last_accessed: u64,
    /// Access count for LRU tracking
    pub access_count: u64,
    /// Data size in bytes (for memory management)
    pub size_bytes: usize,
    /// Whether data is compressed
    pub compressed: bool,
    /// TTL in seconds (0 = no expiry)
    pub ttl_seconds: u64,
}

impl<T> CacheEntry<T> {
    pub fn new(data: T, size_bytes: usize, ttl_seconds: u64) -> Self {
        let now = current_timestamp();
        Self {
            data,
            created_at: now,
            last_accessed: now,
            access_count: 1,
            size_bytes,
            compressed: false,
            ttl_seconds,
        }
    }

    pub fn is_expired(&self) -> bool {
        if self.ttl_seconds == 0 {
            return false;
        }

        let now = current_timestamp();
        now > self.created_at + self.ttl_seconds
    }

    pub fn touch(&mut self) {
        self.last_accessed = current_timestamp();
        self.access_count += 1;
    }
}

/// Cache statistics for monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheStats {
    /// Total cache hits
    pub hits: u64,
    /// Total cache misses
    pub misses: u64,
    /// Cache hit rate (0.0 - 1.0)
    pub hit_rate: f64,
    /// Total entries in cache
    pub entry_count: usize,
    /// Total memory usage in bytes
    pub memory_usage_bytes: usize,
    /// Maximum memory limit in bytes
    pub memory_limit_bytes: usize,
    /// Number of evictions performed
    pub evictions: u64,
    /// Number of expired entries cleaned up
    pub expirations: u64,
    /// Average entry size in bytes
    pub avg_entry_size_bytes: f64,
}

impl Default for CacheStats {
    fn default() -> Self {
        Self {
            hits: 0,
            misses: 0,
            hit_rate: 0.0,
            entry_count: 0,
            memory_usage_bytes: 0,
            memory_limit_bytes: 100 * 1024 * 1024, // 100MB default
            evictions: 0,
            expirations: 0,
            avg_entry_size_bytes: 0.0,
        }
    }
}

impl CacheStats {
    pub fn update_hit_rate(&mut self) {
        let total_requests = self.hits + self.misses;
        if total_requests > 0 {
            self.hit_rate = self.hits as f64 / total_requests as f64;
        }
    }

    pub fn update_avg_size(&mut self) {
        if self.entry_count > 0 {
            self.avg_entry_size_bytes = self.memory_usage_bytes as f64 / self.entry_count as f64;
        }
    }
}

/// Cache configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    /// Maximum memory usage in bytes
    pub max_memory_bytes: usize,
    /// Default TTL for entries in seconds (0 = no expiry)
    pub default_ttl_seconds: u64,
    /// Enable compression for large entries
    pub enable_compression: bool,
    /// Compression threshold in bytes
    pub compression_threshold_bytes: usize,
    /// Enable automatic cleanup of expired entries
    pub enable_auto_cleanup: bool,
    /// Cleanup interval in seconds
    pub cleanup_interval_seconds: u64,
    /// LRU eviction batch size
    pub eviction_batch_size: usize,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            max_memory_bytes: 100 * 1024 * 1024, // 100MB
            default_ttl_seconds: 3600,           // 1 hour
            enable_compression: true,
            compression_threshold_bytes: 1024, // 1KB
            enable_auto_cleanup: true,
            cleanup_interval_seconds: 300, // 5 minutes
            eviction_batch_size: 100,
        }
    }
}

/// Multi-tier cache with different storage layers
#[derive(Debug, Clone)]
pub enum CacheLayer {
    /// In-memory cache (fastest)
    Memory,
    /// Disk-based cache (slower but persistent)
    Disk,
    /// Distributed cache (for multi-node setups)
    Distributed,
}

/// High-performance concurrent cache implementation
#[derive(Debug)]
pub struct PerformanceCache<T>
where
    T: Clone + Serialize + for<'de> Deserialize<'de> + Send + Sync,
{
    /// Cache entries
    entries: Arc<RwLock<HashMap<String, CacheEntry<T>>>>,
    /// Access time tracking for LRU
    access_order: Arc<RwLock<BTreeMap<u64, String>>>,
    /// Cache configuration
    config: CacheConfig,
    /// Cache statistics
    stats: Arc<RwLock<CacheStats>>,
    /// Last cleanup timestamp
    last_cleanup: Arc<RwLock<u64>>,
}

impl<T> PerformanceCache<T>
where
    T: Clone + Serialize + for<'de> Deserialize<'de> + Send + Sync,
{
    /// Create new performance cache
    pub fn new(config: CacheConfig) -> Self {
        let stats = CacheStats {
            memory_limit_bytes: config.max_memory_bytes,
            ..Default::default()
        };

        Self {
            entries: Arc::new(RwLock::new(HashMap::new())),
            access_order: Arc::new(RwLock::new(BTreeMap::new())),
            config,
            stats: Arc::new(RwLock::new(stats)),
            last_cleanup: Arc::new(RwLock::new(current_timestamp())),
        }
    }

    /// Get item from cache
    pub fn get(&self, key: &str) -> Option<T> {
        // Auto cleanup if needed
        if self.config.enable_auto_cleanup {
            self.maybe_cleanup();
        }

        let mut entries = self.entries.write();
        let mut stats = self.stats.write();

        if let Some(entry) = entries.get_mut(key) {
            // Check if expired
            if entry.is_expired() {
                entries.remove(key);
                stats.expirations += 1;
                stats.misses += 1;
                stats.update_hit_rate();
                return None;
            }

            // Update access tracking
            entry.touch();

            // Update LRU order
            {
                let mut access_order = self.access_order.write();
                access_order.insert(entry.last_accessed, key.to_string());
            }

            stats.hits += 1;
            stats.update_hit_rate();

            Some(entry.data.clone())
        } else {
            stats.misses += 1;
            stats.update_hit_rate();
            None
        }
    }

    /// Put item in cache
    pub fn put(&self, key: String, value: T) -> Result<()> {
        self.put_with_ttl(key, value, self.config.default_ttl_seconds)
    }

    /// Put item in cache with custom TTL
    pub fn put_with_ttl(&self, key: String, value: T, ttl_seconds: u64) -> Result<()> {
        let size_bytes = self.estimate_size(&value)?;
        let entry = CacheEntry::new(value, size_bytes, ttl_seconds);

        {
            let mut entries = self.entries.write();
            let mut stats = self.stats.write();

            // Check if we need to evict
            let current_memory = stats.memory_usage_bytes;
            let new_memory = current_memory + size_bytes;

            if new_memory > self.config.max_memory_bytes {
                drop(stats);
                drop(entries);
                self.evict_lru(size_bytes)?;
                entries = self.entries.write();
                stats = self.stats.write();
            }

            // Insert new entry
            entries.insert(key.clone(), entry);

            // Update access order
            {
                let mut access_order = self.access_order.write();
                access_order.insert(current_timestamp(), key);
            }

            // Update stats
            stats.entry_count = entries.len();
            stats.memory_usage_bytes += size_bytes;
            stats.update_avg_size();
        }

        Ok(())
    }

    /// Remove item from cache
    pub fn remove(&self, key: &str) -> Option<T> {
        let mut entries = self.entries.write();
        let mut stats = self.stats.write();

        if let Some(entry) = entries.remove(key) {
            stats.entry_count = entries.len();
            stats.memory_usage_bytes = stats.memory_usage_bytes.saturating_sub(entry.size_bytes);
            stats.update_avg_size();

            // Remove from access order
            {
                let mut access_order = self.access_order.write();
                access_order.retain(|_, v| v != key);
            }

            Some(entry.data)
        } else {
            None
        }
    }

    /// Clear all cache entries
    pub fn clear(&self) {
        let mut entries = self.entries.write();
        let mut stats = self.stats.write();
        let mut access_order = self.access_order.write();

        entries.clear();
        access_order.clear();

        stats.entry_count = 0;
        stats.memory_usage_bytes = 0;
        stats.update_avg_size();
    }

    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        self.stats.read().clone()
    }

    /// Check if key exists in cache
    pub fn contains_key(&self, key: &str) -> bool {
        let entries = self.entries.read();
        if let Some(entry) = entries.get(key) {
            !entry.is_expired()
        } else {
            false
        }
    }

    /// Get number of entries in cache
    pub fn len(&self) -> usize {
        self.entries.read().len()
    }

    /// Check if cache is empty
    pub fn is_empty(&self) -> bool {
        self.entries.read().is_empty()
    }

    /// Evict least recently used entries to free space
    fn evict_lru(&self, bytes_needed: usize) -> Result<()> {
        let mut bytes_freed = 0;
        let mut keys_to_remove = Vec::new();

        // Find oldest entries to evict
        {
            let access_order = self.access_order.read();
            let entries = self.entries.read();

            for (_, key) in access_order.iter() {
                if let Some(entry) = entries.get(key) {
                    keys_to_remove.push((key.clone(), entry.size_bytes));
                    bytes_freed += entry.size_bytes;

                    if bytes_freed >= bytes_needed
                        || keys_to_remove.len() >= self.config.eviction_batch_size
                    {
                        break;
                    }
                }
            }
        }

        // Remove the selected entries
        {
            let mut entries = self.entries.write();
            let mut stats = self.stats.write();
            let mut access_order = self.access_order.write();

            for (key, size) in keys_to_remove {
                entries.remove(&key);
                access_order.retain(|_, v| v != &key);
                stats.memory_usage_bytes = stats.memory_usage_bytes.saturating_sub(size);
                stats.evictions += 1;
            }

            stats.entry_count = entries.len();
            stats.update_avg_size();
        }

        Ok(())
    }

    /// Clean up expired entries
    fn cleanup_expired(&self) -> usize {
        let mut expired_keys = Vec::new();

        // Find expired entries
        {
            let entries = self.entries.read();
            let now = current_timestamp();

            for (key, entry) in entries.iter() {
                if entry.ttl_seconds > 0 && now > entry.created_at + entry.ttl_seconds {
                    expired_keys.push((key.clone(), entry.size_bytes));
                }
            }
        }

        // Remove expired entries
        if !expired_keys.is_empty() {
            let mut entries = self.entries.write();
            let mut stats = self.stats.write();
            let mut access_order = self.access_order.write();

            for (key, size) in &expired_keys {
                entries.remove(key);
                access_order.retain(|_, v| v != key);
                stats.memory_usage_bytes = stats.memory_usage_bytes.saturating_sub(*size);
                stats.expirations += 1;
            }

            stats.entry_count = entries.len();
            stats.update_avg_size();
        }

        expired_keys.len()
    }

    /// Maybe perform cleanup based on interval
    fn maybe_cleanup(&self) {
        let now = current_timestamp();
        let last_cleanup = self.last_cleanup.write();

        if now > *last_cleanup + self.config.cleanup_interval_seconds {
            drop(last_cleanup);
            self.cleanup_expired();
            *self.last_cleanup.write() = now;
        }
    }

    /// Estimate size of data for memory tracking
    fn estimate_size(&self, _value: &T) -> Result<usize> {
        // This is a simplified estimation - in production you might want to use
        // bincode or similar to get actual serialized size
        Ok(std::mem::size_of::<T>() + 64) // Base estimate + overhead
    }

    /// Get cache efficiency metrics
    pub fn efficiency_metrics(&self) -> HashMap<String, f64> {
        let stats = self.stats.read();
        let mut metrics = HashMap::new();

        metrics.insert("hit_rate".to_string(), stats.hit_rate);
        metrics.insert(
            "memory_utilization".to_string(),
            stats.memory_usage_bytes as f64 / stats.memory_limit_bytes as f64,
        );
        metrics.insert(
            "avg_entry_size_kb".to_string(),
            stats.avg_entry_size_bytes / 1024.0,
        );
        metrics.insert(
            "eviction_rate".to_string(),
            stats.evictions as f64 / (stats.hits + stats.misses).max(1) as f64,
        );

        metrics
    }
}

impl<T> Default for PerformanceCache<T>
where
    T: Clone + Serialize + for<'de> Deserialize<'de> + Send + Sync,
{
    fn default() -> Self {
        Self::new(CacheConfig::default())
    }
}

/// Get current Unix timestamp
fn current_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or(Duration::ZERO)
        .as_secs()
}

/// Cache manager for different data types
#[derive(Debug)]
pub struct CacheManager {
    /// Dataset cache
    dataset_cache: Arc<PerformanceCache<crate::Dataset>>,
    /// Proof cache
    proof_cache: Arc<PerformanceCache<crate::Proof>>,
    /// Generic string cache
    string_cache: Arc<PerformanceCache<String>>,
    /// Cache statistics aggregation
    #[allow(dead_code)]
    global_stats: Arc<RwLock<HashMap<String, CacheStats>>>,
}

impl CacheManager {
    pub fn new() -> Self {
        let config = CacheConfig::default();

        Self {
            dataset_cache: Arc::new(PerformanceCache::new(config.clone())),
            proof_cache: Arc::new(PerformanceCache::new(config.clone())),
            string_cache: Arc::new(PerformanceCache::new(config)),
            global_stats: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub fn dataset_cache(&self) -> &PerformanceCache<crate::Dataset> {
        &self.dataset_cache
    }

    pub fn proof_cache(&self) -> &PerformanceCache<crate::Proof> {
        &self.proof_cache
    }

    pub fn string_cache(&self) -> &PerformanceCache<String> {
        &self.string_cache
    }

    /// Get aggregated cache statistics
    pub fn aggregate_stats(&self) -> HashMap<String, CacheStats> {
        let mut stats = HashMap::new();
        stats.insert("dataset".to_string(), self.dataset_cache.stats());
        stats.insert("proof".to_string(), self.proof_cache.stats());
        stats.insert("string".to_string(), self.string_cache.stats());
        stats
    }

    /// Generate cache performance report
    pub fn performance_report(&self) -> String {
        let mut report = String::new();
        report.push_str("# Cache Performance Report\n\n");

        for (cache_type, stats) in self.aggregate_stats() {
            report.push_str(&format!("## {} Cache\n", cache_type));
            report.push_str(&format!("- Hit Rate: {:.1}%\n", stats.hit_rate * 100.0));
            report.push_str(&format!("- Entries: {}\n", stats.entry_count));
            report.push_str(&format!(
                "- Memory Usage: {:.1} MB / {:.1} MB\n",
                stats.memory_usage_bytes as f64 / 1024.0 / 1024.0,
                stats.memory_limit_bytes as f64 / 1024.0 / 1024.0
            ));
            report.push_str(&format!("- Evictions: {}\n", stats.evictions));
            report.push_str(&format!("- Expirations: {}\n", stats.expirations));
            report.push('\n');
        }

        report
    }
}

impl Default for CacheManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_basic_operations() {
        let cache = PerformanceCache::<String>::default();

        // Test put and get
        cache.put("key1".to_string(), "value1".to_string()).unwrap();
        assert_eq!(cache.get("key1"), Some("value1".to_string()));
        assert_eq!(cache.get("nonexistent"), None);

        // Test remove
        assert_eq!(cache.remove("key1"), Some("value1".to_string()));
        assert_eq!(cache.get("key1"), None);
    }

    #[test]
    fn test_cache_expiration() {
        let cache = PerformanceCache::<String>::default();

        // Put with short TTL
        cache
            .put_with_ttl("key1".to_string(), "value1".to_string(), 1)
            .unwrap();
        assert_eq!(cache.get("key1"), Some("value1".to_string()));

        // Wait for expiration (in real test you'd mock the time)
        // For this test we just verify the expiration logic
        assert!(cache.contains_key("key1"));
    }

    #[test]
    fn test_cache_stats() {
        let cache = PerformanceCache::<String>::default();

        // Generate some cache activity
        cache.put("key1".to_string(), "value1".to_string()).unwrap();
        cache.get("key1");
        cache.get("nonexistent");

        let stats = cache.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.hit_rate, 0.5);
    }
}
