//! Storage backends for the ZKP Dataset Ledger.

use crate::error::LedgerError;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Trait for storage backend implementations.
pub trait Storage: Send + Sync {
    /// Store a key-value pair.
    fn put(&mut self, key: &str, value: &[u8]) -> Result<(), LedgerError>;

    /// Retrieve a value by key.
    fn get(&self, key: &str) -> Result<Option<Vec<u8>>, LedgerError>;

    /// Delete a key-value pair.
    fn delete(&mut self, key: &str) -> Result<(), LedgerError>;

    /// List all keys with a given prefix.
    fn list_keys(&self, prefix: &str) -> Result<Vec<String>, LedgerError>;

    /// Check if a key exists.
    fn exists(&self, key: &str) -> Result<bool, LedgerError>;

    /// Get storage statistics.
    fn stats(&self) -> Result<StorageStats, LedgerError>;
}

/// Storage statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageStats {
    pub total_keys: usize,
    pub total_size_bytes: u64,
    pub backend_type: String,
}

/// In-memory storage backend for testing.
#[derive(Debug, Clone, Default)]
pub struct MemoryStorage {
    data: HashMap<String, Vec<u8>>,
}

impl MemoryStorage {
    pub fn new() -> Self {
        Self {
            data: HashMap::new(),
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            data: HashMap::with_capacity(capacity),
        }
    }
}

impl Storage for MemoryStorage {
    fn put(&mut self, key: &str, value: &[u8]) -> Result<(), LedgerError> {
        self.data.insert(key.to_string(), value.to_vec());
        Ok(())
    }

    fn get(&self, key: &str) -> Result<Option<Vec<u8>>, LedgerError> {
        Ok(self.data.get(key).cloned())
    }

    fn delete(&mut self, key: &str) -> Result<(), LedgerError> {
        self.data.remove(key);
        Ok(())
    }

    fn list_keys(&self, prefix: &str) -> Result<Vec<String>, LedgerError> {
        Ok(self
            .data
            .keys()
            .filter(|k| k.starts_with(prefix))
            .cloned()
            .collect())
    }

    fn exists(&self, key: &str) -> Result<bool, LedgerError> {
        Ok(self.data.contains_key(key))
    }

    fn stats(&self) -> Result<StorageStats, LedgerError> {
        let total_size_bytes = self.data.values().map(|v| v.len() as u64).sum();

        Ok(StorageStats {
            total_keys: self.data.len(),
            total_size_bytes,
            backend_type: "memory".to_string(),
        })
    }
}

/// RocksDB storage backend for high-performance persistent storage.
#[cfg(feature = "rocksdb")]
pub mod rocksdb_backend {
    use super::{Storage, StorageStats};
    use crate::error::LedgerError;
    use rocksdb::{Options, DB};
    use std::path::Path;
    use std::sync::Arc;

    /// RocksDB storage implementation
    pub struct RocksDBStorage {
        db: Arc<DB>,
        path: String,
    }

    impl RocksDBStorage {
        /// Create a new RocksDB storage instance
        pub fn new<P: AsRef<Path>>(path: P) -> Result<Self, LedgerError> {
            let mut opts = Options::default();
            opts.create_if_missing(true);
            opts.set_compression_type(rocksdb::DBCompressionType::Zstd);

            // Performance optimizations
            opts.set_max_background_jobs(6);
            opts.set_write_buffer_size(128 * 1024 * 1024); // 128MB
            opts.set_target_file_size_base(64 * 1024 * 1024); // 64MB
            opts.set_level_zero_file_num_compaction_trigger(4);
            opts.set_level_zero_slowdown_writes_trigger(20);
            opts.set_level_zero_stop_writes_trigger(36);

            // Enable bloom filters for faster reads
            let mut block_opts = rocksdb::BlockBasedOptions::default();
            block_opts.set_bloom_filter(10.0, false);
            block_opts.set_cache_index_and_filter_blocks(true);
            opts.set_block_based_table_factory(&block_opts);

            let path_str = path.as_ref().to_string_lossy().to_string();
            let db = DB::open(&opts, &path_str)
                .map_err(|e| LedgerError::StorageError(format!("Failed to open RocksDB: {}", e)))?;

            Ok(RocksDBStorage {
                db: Arc::new(db),
                path: path_str,
            })
        }

        /// Compact the database to optimize storage
        pub fn compact(&self) -> Result<(), LedgerError> {
            self.db.compact_range(None::<&[u8]>, None::<&[u8]>);
            Ok(())
        }

        /// Get database statistics
        pub fn get_property(&self, property: &str) -> Result<Option<String>, LedgerError> {
            Ok(self
                .db
                .property_value(property)
                .map_err(|e| LedgerError::StorageError(format!("Property query failed: {}", e)))?)
        }
    }

    impl Storage for RocksDBStorage {
        fn put(&mut self, key: &str, value: &[u8]) -> Result<(), LedgerError> {
            self.db
                .put(key.as_bytes(), value)
                .map_err(|e| LedgerError::StorageError(format!("Put operation failed: {}", e)))
        }

        fn get(&self, key: &str) -> Result<Option<Vec<u8>>, LedgerError> {
            self.db
                .get(key.as_bytes())
                .map_err(|e| LedgerError::StorageError(format!("Get operation failed: {}", e)))
        }

        fn delete(&mut self, key: &str) -> Result<(), LedgerError> {
            self.db
                .delete(key.as_bytes())
                .map_err(|e| LedgerError::StorageError(format!("Delete operation failed: {}", e)))
        }

        fn list_keys(&self, prefix: &str) -> Result<Vec<String>, LedgerError> {
            let mut keys = Vec::new();
            let iter = self.db.prefix_iterator(prefix.as_bytes());

            for item in iter {
                match item {
                    Ok((key, _)) => {
                        if let Ok(key_str) = String::from_utf8(key.to_vec()) {
                            if key_str.starts_with(prefix) {
                                keys.push(key_str);
                            } else {
                                break; // Prefix iteration finished
                            }
                        }
                    }
                    Err(e) => {
                        return Err(LedgerError::StorageError(format!("Iterator error: {}", e)));
                    }
                }
            }

            Ok(keys)
        }

        fn exists(&self, key: &str) -> Result<bool, LedgerError> {
            match self.get(key)? {
                Some(_) => Ok(true),
                None => Ok(false),
            }
        }

        fn stats(&self) -> Result<StorageStats, LedgerError> {
            // Get approximate key count
            let num_keys = self
                .get_property("rocksdb.estimate-num-keys")?
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(0);

            // Get approximate size
            let size_bytes = self
                .get_property("rocksdb.total-sst-files-size")?
                .and_then(|s| s.parse::<u64>().ok())
                .unwrap_or(0);

            Ok(StorageStats {
                total_keys: num_keys,
                total_size_bytes: size_bytes,
                backend_type: "rocksdb".to_string(),
            })
        }
    }
}

#[cfg(feature = "rocksdb")]
pub use rocksdb_backend::RocksDBStorage;

/// PostgreSQL storage backend.
#[cfg(feature = "postgres")]
pub mod postgres_backend {
    use super::*;
    use sqlx::{postgres::PgPoolOptions, PgPool, Row};
    use tokio::runtime::Runtime;

    pub struct PostgresStorage {
        pool: PgPool,
        rt: Runtime,
    }

    impl PostgresStorage {
        pub async fn new(database_url: &str) -> Result<Self, LedgerError> {
            let pool = PgPoolOptions::new()
                .max_connections(10)
                .connect(database_url)
                .await
                .map_err(|e| {
                    LedgerError::StorageError(format!("PostgreSQL connection failed: {}", e))
                })?;

            // Create table if it doesn't exist
            sqlx::query(
                r#"
                CREATE TABLE IF NOT EXISTS ledger_storage (
                    key TEXT PRIMARY KEY,
                    value BYTEA NOT NULL,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                )
                "#,
            )
            .execute(&pool)
            .await
            .map_err(|e| LedgerError::StorageError(format!("Table creation failed: {}", e)))?;

            let rt = Runtime::new().map_err(|e| {
                LedgerError::StorageError(format!("Runtime creation failed: {}", e))
            })?;

            Ok(Self { pool, rt })
        }
    }

    impl Storage for PostgresStorage {
        fn put(&mut self, key: &str, value: &[u8]) -> Result<(), LedgerError> {
            self.rt.block_on(async {
                sqlx::query(
                    r#"
                    INSERT INTO ledger_storage (key, value, updated_at)
                    VALUES ($1, $2, NOW())
                    ON CONFLICT (key) DO UPDATE SET
                        value = EXCLUDED.value,
                        updated_at = NOW()
                    "#,
                )
                .bind(key)
                .bind(value)
                .execute(&self.pool)
                .await
                .map_err(|e| LedgerError::StorageError(format!("PostgreSQL put failed: {}", e)))?;

                Ok(())
            })
        }

        fn get(&self, key: &str) -> Result<Option<Vec<u8>>, LedgerError> {
            self.rt.block_on(async {
                let row = sqlx::query("SELECT value FROM ledger_storage WHERE key = $1")
                    .bind(key)
                    .fetch_optional(&self.pool)
                    .await
                    .map_err(|e| {
                        LedgerError::StorageError(format!("PostgreSQL get failed: {}", e))
                    })?;

                Ok(row.map(|r| r.get::<Vec<u8>, _>("value")))
            })
        }

        fn delete(&mut self, key: &str) -> Result<(), LedgerError> {
            self.rt.block_on(async {
                sqlx::query("DELETE FROM ledger_storage WHERE key = $1")
                    .bind(key)
                    .execute(&self.pool)
                    .await
                    .map_err(|e| {
                        LedgerError::StorageError(format!("PostgreSQL delete failed: {}", e))
                    })?;

                Ok(())
            })
        }

        fn list_keys(&self, prefix: &str) -> Result<Vec<String>, LedgerError> {
            self.rt.block_on(async {
                let rows = sqlx::query("SELECT key FROM ledger_storage WHERE key LIKE $1")
                    .bind(format!("{}%", prefix))
                    .fetch_all(&self.pool)
                    .await
                    .map_err(|e| {
                        LedgerError::StorageError(format!("PostgreSQL list_keys failed: {}", e))
                    })?;

                Ok(rows
                    .into_iter()
                    .map(|row| row.get::<String, _>("key"))
                    .collect())
            })
        }

        fn exists(&self, key: &str) -> Result<bool, LedgerError> {
            self.rt.block_on(async {
                let count: i64 =
                    sqlx::query_scalar("SELECT COUNT(*) FROM ledger_storage WHERE key = $1")
                        .bind(key)
                        .fetch_one(&self.pool)
                        .await
                        .map_err(|e| {
                            LedgerError::StorageError(format!(
                                "PostgreSQL exists check failed: {}",
                                e
                            ))
                        })?;

                Ok(count > 0)
            })
        }

        fn stats(&self) -> Result<StorageStats, LedgerError> {
            self.rt.block_on(async {
                let (total_keys, total_size_bytes): (i64, i64) = sqlx::query_as(
                    "SELECT COUNT(*), COALESCE(SUM(octet_length(key) + octet_length(value)), 0) FROM ledger_storage"
                )
                .fetch_one(&self.pool)
                .await
                .map_err(|e| LedgerError::StorageError(format!("PostgreSQL stats failed: {}", e)))?;
                Ok(StorageStats {
                    total_keys: total_keys as usize,
                    total_size_bytes: total_size_bytes as u64,
                    backend_type: "postgres".to_string(),
                })
            })
        }
    }
}

/// Create a storage backend based on configuration.
pub fn create_storage(backend_type: &str, config: &str) -> Result<Box<dyn Storage>, LedgerError> {
    match backend_type {
        "memory" => Ok(Box::new(MemoryStorage::new())),

        #[cfg(feature = "rocksdb")]
        "rocksdb" => {
            use rocksdb_backend::RocksDBStorage;
            let path = if config.is_empty() {
                "./zkp_ledger.db"
            } else {
                config
            };
            let storage = RocksDBStorage::new(path)?;
            Ok(Box::new(storage))
        }

        #[cfg(feature = "postgres")]
        "postgres" => {
            use postgres_backend::PostgresStorage;
            let rt = tokio::runtime::Runtime::new().map_err(|e| {
                LedgerError::StorageError(format!("Runtime creation failed: {}", e))
            })?;
            let storage = rt.block_on(PostgresStorage::new(config))?;
            Ok(Box::new(storage))
        }

        _ => Err(LedgerError::StorageError(format!(
            "Unknown storage backend: {}",
            backend_type
        ))),
    }
}

/// Storage transaction for atomic operations.
pub struct Transaction<'a> {
    storage: &'a mut dyn Storage,
    operations: Vec<Operation>,
    committed: bool,
}

#[derive(Debug, Clone)]
enum Operation {
    Put { key: String, value: Vec<u8> },
    Delete { key: String },
}

impl<'a> Transaction<'a> {
    pub fn new(storage: &'a mut dyn Storage) -> Self {
        Self {
            storage,
            operations: Vec::new(),
            committed: false,
        }
    }

    pub fn put(&mut self, key: &str, value: &[u8]) {
        self.operations.push(Operation::Put {
            key: key.to_string(),
            value: value.to_vec(),
        });
    }

    pub fn delete(&mut self, key: &str) {
        self.operations.push(Operation::Delete {
            key: key.to_string(),
        });
    }

    pub fn commit(mut self) -> Result<(), LedgerError> {
        for op in &self.operations {
            match op {
                Operation::Put { key, value } => {
                    self.storage.put(key, value)?;
                }
                Operation::Delete { key } => {
                    self.storage.delete(key)?;
                }
            }
        }

        self.committed = true;
        Ok(())
    }
}

impl<'a> Drop for Transaction<'a> {
    fn drop(&mut self) {
        if !self.committed {
            // In a real implementation, we would rollback here
            eprintln!("Warning: Transaction dropped without commit");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_storage() {
        let mut storage = MemoryStorage::new();

        // Test put/get
        storage.put("key1", b"value1").unwrap();
        assert_eq!(storage.get("key1").unwrap(), Some(b"value1".to_vec()));

        // Test exists
        assert!(storage.exists("key1").unwrap());
        assert!(!storage.exists("key2").unwrap());

        // Test list_keys
        storage.put("prefix_a", b"value_a").unwrap();
        storage.put("prefix_b", b"value_b").unwrap();
        storage.put("other", b"value_other").unwrap();

        let keys = storage.list_keys("prefix").unwrap();
        assert_eq!(keys.len(), 2);
        assert!(keys.contains(&"prefix_a".to_string()));
        assert!(keys.contains(&"prefix_b".to_string()));

        // Test delete
        storage.delete("key1").unwrap();
        assert_eq!(storage.get("key1").unwrap(), None);

        // Test stats
        let stats = storage.stats().unwrap();
        assert_eq!(stats.total_keys, 3);
        assert_eq!(stats.backend_type, "memory");
    }

    #[test]
    fn test_transaction() {
        let mut storage = MemoryStorage::new();

        {
            let mut tx = Transaction::new(&mut storage);
            tx.put("key1", b"value1");
            tx.put("key2", b"value2");
            tx.commit().unwrap();
        }

        assert_eq!(storage.get("key1").unwrap(), Some(b"value1".to_vec()));
        assert_eq!(storage.get("key2").unwrap(), Some(b"value2".to_vec()));
    }
}
