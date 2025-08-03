//! Storage backends for the ZKP Dataset Ledger.

use crate::error::LedgerError;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

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

// RocksDB storage backend temporarily disabled due to build dependencies
// #[cfg(feature = "rocksdb")]
// pub mod rocksdb_backend { ... }

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
