use tempfile::TempDir;
use uuid::Uuid;
use zkp_dataset_ledger::{
    storage::{MemoryStorage, Storage, StorageStats},
    LedgerError, Result,
};

#[cfg(feature = "rocksdb")]
use zkp_dataset_ledger::storage::RocksDBStorage;

#[tokio::test]
async fn test_memory_storage_basic_operations() -> Result<()> {
    let mut storage = MemoryStorage::new();

    // Test put and get
    let key = "test_key";
    let value = b"test_value";

    storage.put(key, value)?;

    let retrieved = storage.get(key)?;
    assert!(retrieved.is_some());
    assert_eq!(retrieved.unwrap(), value);

    // Test delete
    assert!(storage.delete(key)?);
    assert!(storage.get(key)?.is_none());

    // Test delete non-existent key
    assert!(!storage.delete("non_existent")?);

    Ok(())
}

#[tokio::test]
async fn test_memory_storage_list_keys() -> Result<()> {
    let mut storage = MemoryStorage::new();

    // Add multiple keys with prefix
    let keys = vec!["prefix:key1", "prefix:key2", "prefix:key3", "other:key1"];

    for key in &keys {
        storage.put(key, b"value")?;
    }

    // List keys with prefix
    let prefix_keys = storage.list_keys("prefix:")?;
    assert_eq!(prefix_keys.len(), 3);
    for key in prefix_keys {
        assert!(key.starts_with("prefix:"));
    }

    // List all keys
    let all_keys = storage.list_keys("")?;
    assert_eq!(all_keys.len(), 4);

    Ok(())
}

#[tokio::test]
async fn test_memory_storage_stats() -> Result<()> {
    let mut storage = MemoryStorage::new();

    // Initially empty
    let stats = storage.stats()?;
    assert_eq!(stats.key_count, 0);
    assert_eq!(stats.total_size_bytes, 0);

    // Add some data
    storage.put("key1", b"value1")?;
    storage.put("key2", b"longer_value_here")?;

    let stats = storage.stats()?;
    assert_eq!(stats.key_count, 2);
    assert!(stats.total_size_bytes > 0);

    Ok(())
}

#[tokio::test]
async fn test_memory_storage_batch_operations() -> Result<()> {
    let mut storage = MemoryStorage::new();

    // Test batch put
    let batch_data = vec![
        ("batch1", b"data1".to_vec()),
        ("batch2", b"data2".to_vec()),
        ("batch3", b"data3".to_vec()),
    ];

    storage.batch_put(&batch_data)?;

    // Verify all data was stored
    for (key, expected_value) in &batch_data {
        let stored_value = storage.get(key)?;
        assert!(stored_value.is_some());
        assert_eq!(stored_value.unwrap(), *expected_value);
    }

    // Test batch delete
    let delete_keys = vec!["batch1", "batch3"];
    storage.batch_delete(&delete_keys)?;

    assert!(storage.get("batch1")?.is_none());
    assert!(storage.get("batch2")?.is_some());
    assert!(storage.get("batch3")?.is_none());

    Ok(())
}

#[tokio::test]
async fn test_memory_storage_concurrent_access() -> Result<()> {
    use std::sync::Arc;
    use tokio::sync::Mutex;

    let storage = Arc::new(Mutex::new(MemoryStorage::new()));
    let mut handles = vec![];

    // Spawn multiple tasks that write to storage
    for i in 0..10 {
        let storage_clone = Arc::clone(&storage);
        let handle = tokio::spawn(async move {
            let mut storage = storage_clone.lock().await;
            let key = format!("concurrent_key_{}", i);
            let value = format!("value_{}", i).into_bytes();
            storage.put(&key, &value).unwrap();
        });
        handles.push(handle);
    }

    // Wait for all tasks to complete
    for handle in handles {
        handle.await.unwrap();
    }

    // Verify all data was written
    let storage = storage.lock().await;
    for i in 0..10 {
        let key = format!("concurrent_key_{}", i);
        assert!(storage.get(&key).unwrap().is_some());
    }

    Ok(())
}

#[cfg(feature = "rocksdb")]
#[tokio::test]
async fn test_rocksdb_storage_basic_operations() -> Result<()> {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let db_path = temp_dir.path().join("test_db");

    {
        let mut storage = RocksDBStorage::new(db_path.to_str().unwrap())?;

        // Test put and get
        let key = "rocksdb_test_key";
        let value = b"rocksdb_test_value";

        storage.put(key, value)?;

        let retrieved = storage.get(key)?;
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap(), value);

        // Test delete
        assert!(storage.delete(key)?);
        assert!(storage.get(key)?.is_none());
    }

    // Test persistence by reopening database
    {
        let mut storage = RocksDBStorage::new(db_path.to_str().unwrap())?;

        storage.put("persistent_key", b"persistent_value")?;

        // Close and reopen
        drop(storage);

        let storage = RocksDBStorage::new(db_path.to_str().unwrap())?;
        let retrieved = storage.get("persistent_key")?;
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap(), b"persistent_value");
    }

    Ok(())
}

#[cfg(feature = "rocksdb")]
#[tokio::test]
async fn test_rocksdb_storage_list_and_stats() -> Result<()> {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let db_path = temp_dir.path().join("test_stats_db");
    let mut storage = RocksDBStorage::new(db_path.to_str().unwrap())?;

    // Add test data
    let test_data = vec![
        ("ledger:entry1", b"data1"),
        ("ledger:entry2", b"data2"),
        ("config:setting1", b"value1"),
        ("config:setting2", b"value2"),
    ];

    for (key, value) in &test_data {
        storage.put(key, value)?;
    }

    // Test list_keys with prefix
    let ledger_keys = storage.list_keys("ledger:")?;
    assert_eq!(ledger_keys.len(), 2);

    let config_keys = storage.list_keys("config:")?;
    assert_eq!(config_keys.len(), 2);

    // Test stats
    let stats = storage.stats()?;
    assert_eq!(stats.key_count, 4);
    assert!(stats.total_size_bytes > 0);

    Ok(())
}

#[cfg(feature = "rocksdb")]
#[tokio::test]
async fn test_rocksdb_storage_batch_operations() -> Result<()> {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let db_path = temp_dir.path().join("test_batch_db");
    let mut storage = RocksDBStorage::new(db_path.to_str().unwrap())?;

    // Test batch put
    let batch_data = vec![
        ("batch_key1", b"batch_value1".to_vec()),
        ("batch_key2", b"batch_value2".to_vec()),
        ("batch_key3", b"batch_value3".to_vec()),
    ];

    storage.batch_put(&batch_data)?;

    // Verify all data was stored
    for (key, expected_value) in &batch_data {
        let stored_value = storage.get(key)?;
        assert!(stored_value.is_some());
        assert_eq!(stored_value.unwrap(), *expected_value);
    }

    // Test batch delete
    let delete_keys = vec!["batch_key1", "batch_key3"];
    storage.batch_delete(&delete_keys)?;

    assert!(storage.get("batch_key1")?.is_none());
    assert!(storage.get("batch_key2")?.is_some());
    assert!(storage.get("batch_key3")?.is_none());

    Ok(())
}

#[tokio::test]
async fn test_storage_error_handling() -> Result<()> {
    let mut storage = MemoryStorage::new();

    // Test getting non-existent key
    assert!(storage.get("non_existent")?.is_none());

    // Test deleting non-existent key
    assert!(!storage.delete("non_existent")?);

    // Test listing with non-matching prefix
    let keys = storage.list_keys("no_match:")?;
    assert!(keys.is_empty());

    Ok(())
}

#[tokio::test]
async fn test_storage_large_data() -> Result<()> {
    let mut storage = MemoryStorage::new();

    // Test with large data (1MB)
    let large_data = vec![0u8; 1024 * 1024];
    storage.put("large_key", &large_data)?;

    let retrieved = storage.get("large_key")?;
    assert!(retrieved.is_some());
    assert_eq!(retrieved.unwrap().len(), large_data.len());

    let stats = storage.stats()?;
    assert!(stats.total_size_bytes >= large_data.len() as u64);

    Ok(())
}

#[tokio::test]
async fn test_storage_unicode_keys_and_values() -> Result<()> {
    let mut storage = MemoryStorage::new();

    // Test with Unicode keys and values
    let unicode_key = "测试键🔑";
    let unicode_value = "测试值📊".as_bytes();

    storage.put(unicode_key, unicode_value)?;

    let retrieved = storage.get(unicode_key)?;
    assert!(retrieved.is_some());
    assert_eq!(retrieved.unwrap(), unicode_value);

    // Test listing works with Unicode
    let keys = storage.list_keys("测试")?;
    assert_eq!(keys.len(), 1);
    assert_eq!(keys[0], unicode_key);

    Ok(())
}

#[tokio::test]
async fn test_storage_edge_cases() -> Result<()> {
    let mut storage = MemoryStorage::new();

    // Test empty key and value
    storage.put("", b"")?;
    let retrieved = storage.get("")?;
    assert!(retrieved.is_some());
    assert!(retrieved.unwrap().is_empty());

    // Test very long key
    let long_key = "x".repeat(1000);
    storage.put(&long_key, b"value")?;
    assert!(storage.get(&long_key)?.is_some());

    // Test many small entries
    for i in 0..1000 {
        let key = format!("small_{}", i);
        storage.put(&key, b"x")?;
    }

    let stats = storage.stats()?;
    assert!(stats.key_count >= 1001); // 1000 + empty key + long key

    Ok(())
}

#[tokio::test]
async fn test_storage_prefix_filtering() -> Result<()> {
    let mut storage = MemoryStorage::new();

    // Add keys with various prefixes
    let test_keys = vec![
        "user:alice",
        "user:bob",
        "user:charlie",
        "admin:root",
        "admin:system",
        "data:2024:jan",
        "data:2024:feb",
        "data:2023:dec",
        "temp:session1",
        "temp:session2",
    ];

    for key in &test_keys {
        storage.put(key, b"value")?;
    }

    // Test exact prefix matching
    let user_keys = storage.list_keys("user:")?;
    assert_eq!(user_keys.len(), 3);

    let admin_keys = storage.list_keys("admin:")?;
    assert_eq!(admin_keys.len(), 2);

    // Test nested prefix matching
    let data_2024_keys = storage.list_keys("data:2024:")?;
    assert_eq!(data_2024_keys.len(), 2);

    let data_keys = storage.list_keys("data:")?;
    assert_eq!(data_keys.len(), 3);

    // Test prefix that matches no keys
    let no_match_keys = storage.list_keys("nonexistent:")?;
    assert!(no_match_keys.is_empty());

    Ok(())
}

#[cfg(feature = "rocksdb")]
#[tokio::test]
async fn test_rocksdb_storage_performance() -> Result<()> {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let db_path = temp_dir.path().join("perf_test_db");
    let mut storage = RocksDBStorage::new(db_path.to_str().unwrap())?;

    let start = std::time::Instant::now();

    // Insert 10,000 entries
    for i in 0..10_000 {
        let key = format!("perf_test_{:05}", i);
        let value = format!("value_for_key_{}", i);
        storage.put(&key, value.as_bytes())?;
    }

    let insert_time = start.elapsed();
    println!("RocksDB: Inserted 10,000 entries in {:?}", insert_time);

    let start = std::time::Instant::now();

    // Read all entries back
    for i in 0..10_000 {
        let key = format!("perf_test_{:05}", i);
        let value = storage.get(&key)?;
        assert!(value.is_some());
    }

    let read_time = start.elapsed();
    println!("RocksDB: Read 10,000 entries in {:?}", read_time);

    // Ensure reasonable performance (these are very loose bounds)
    assert!(insert_time.as_secs() < 10, "Insert time too slow");
    assert!(read_time.as_secs() < 5, "Read time too slow");

    Ok(())
}

#[tokio::test]
async fn test_memory_storage_performance() -> Result<()> {
    let mut storage = MemoryStorage::new();

    let start = std::time::Instant::now();

    // Insert 50,000 entries (more than RocksDB test since memory is faster)
    for i in 0..50_000 {
        let key = format!("mem_perf_test_{:05}", i);
        let value = format!("value_for_key_{}", i);
        storage.put(&key, value.as_bytes())?;
    }

    let insert_time = start.elapsed();
    println!("Memory: Inserted 50,000 entries in {:?}", insert_time);

    let start = std::time::Instant::now();

    // Read all entries back
    for i in 0..50_000 {
        let key = format!("mem_perf_test_{:05}", i);
        let value = storage.get(&key)?;
        assert!(value.is_some());
    }

    let read_time = start.elapsed();
    println!("Memory: Read 50,000 entries in {:?}", read_time);

    // Memory storage should be very fast
    assert!(insert_time.as_secs() < 5, "Memory insert time too slow");
    assert!(read_time.as_secs() < 2, "Memory read time too slow");

    Ok(())
}
