//! Unit tests for the storage module.

use zkp_dataset_ledger::error::LedgerError;
use zkp_dataset_ledger::storage::{MemoryStorage, Storage, StorageStats, Transaction};

/// Test basic memory storage operations.
#[test]
fn test_memory_storage_basic() {
    let mut storage = MemoryStorage::new();

    // Test put and get
    storage.put("key1", b"value1").unwrap();
    assert_eq!(storage.get("key1").unwrap(), Some(b"value1".to_vec()));

    // Test non-existent key
    assert_eq!(storage.get("nonexistent").unwrap(), None);

    // Test update
    storage.put("key1", b"updated_value").unwrap();
    assert_eq!(
        storage.get("key1").unwrap(),
        Some(b"updated_value".to_vec())
    );

    // Test delete
    storage.delete("key1").unwrap();
    assert_eq!(storage.get("key1").unwrap(), None);
}

/// Test storage key listing with prefix.
#[test]
fn test_storage_list_keys() {
    let mut storage = MemoryStorage::new();

    // Insert test data
    storage.put("user:1", b"alice").unwrap();
    storage.put("user:2", b"bob").unwrap();
    storage.put("user:3", b"charlie").unwrap();
    storage.put("data:1", b"dataset1").unwrap();
    storage.put("data:2", b"dataset2").unwrap();

    // Test prefix listing
    let user_keys = storage.list_keys("user:").unwrap();
    assert_eq!(user_keys.len(), 3);
    assert!(user_keys.contains(&"user:1".to_string()));
    assert!(user_keys.contains(&"user:2".to_string()));
    assert!(user_keys.contains(&"user:3".to_string()));

    let data_keys = storage.list_keys("data:").unwrap();
    assert_eq!(data_keys.len(), 2);
    assert!(data_keys.contains(&"data:1".to_string()));
    assert!(data_keys.contains(&"data:2".to_string()));

    // Test empty prefix
    let all_keys = storage.list_keys("").unwrap();
    assert_eq!(all_keys.len(), 5);
}

/// Test storage existence checks.
#[test]
fn test_storage_exists() {
    let mut storage = MemoryStorage::new();

    // Test non-existent key
    assert!(!storage.exists("key1").unwrap());

    // Test existent key
    storage.put("key1", b"value1").unwrap();
    assert!(storage.exists("key1").unwrap());

    // Test after deletion
    storage.delete("key1").unwrap();
    assert!(!storage.exists("key1").unwrap());
}

/// Test storage statistics.
#[test]
fn test_storage_stats() {
    let mut storage = MemoryStorage::new();

    // Test empty storage
    let stats = storage.stats().unwrap();
    assert_eq!(stats.total_keys, 0);
    assert_eq!(stats.total_size_bytes, 0);
    assert_eq!(stats.backend_type, "memory");

    // Add some data
    storage.put("key1", b"short").unwrap();
    storage.put("key2", b"longer_value").unwrap();

    let stats = storage.stats().unwrap();
    assert_eq!(stats.total_keys, 2);
    assert_eq!(stats.total_size_bytes, 5 + 12); // "short" + "longer_value"
    assert_eq!(stats.backend_type, "memory");

    // Test after deletion
    storage.delete("key1").unwrap();
    let stats = storage.stats().unwrap();
    assert_eq!(stats.total_keys, 1);
    assert_eq!(stats.total_size_bytes, 12); // Only "longer_value"
}

/// Test binary data storage.
#[test]
fn test_binary_data_storage() {
    let mut storage = MemoryStorage::new();

    // Test binary data
    let binary_data = vec![0u8, 1, 2, 3, 255, 128, 64];
    storage.put("binary_key", &binary_data).unwrap();

    let retrieved = storage.get("binary_key").unwrap();
    assert_eq!(retrieved, Some(binary_data));
}

/// Test large key and value storage.
#[test]
fn test_large_data_storage() {
    let mut storage = MemoryStorage::new();

    // Test large key
    let large_key = "x".repeat(1000);
    storage.put(&large_key, b"value").unwrap();
    assert_eq!(storage.get(&large_key).unwrap(), Some(b"value".to_vec()));

    // Test large value
    let large_value = vec![42u8; 10000];
    storage.put("large_value_key", &large_value).unwrap();
    assert_eq!(storage.get("large_value_key").unwrap(), Some(large_value));
}

/// Test concurrent-like operations (testing edge cases).
#[test]
fn test_storage_edge_cases() {
    let mut storage = MemoryStorage::new();

    // Test empty key
    storage.put("", b"empty_key_value").unwrap();
    assert_eq!(storage.get("").unwrap(), Some(b"empty_key_value".to_vec()));

    // Test empty value
    storage.put("empty_value_key", b"").unwrap();
    assert_eq!(storage.get("empty_value_key").unwrap(), Some(b"".to_vec()));

    // Test overwrite
    storage.put("overwrite", b"original").unwrap();
    storage.put("overwrite", b"new").unwrap();
    assert_eq!(storage.get("overwrite").unwrap(), Some(b"new".to_vec()));

    // Test multiple deletes
    storage.put("delete_test", b"value").unwrap();
    storage.delete("delete_test").unwrap();
    storage.delete("delete_test").unwrap(); // Should not error
    assert_eq!(storage.get("delete_test").unwrap(), None);
}

/// Test storage transaction basic functionality.
#[test]
fn test_storage_transaction() {
    let mut storage = MemoryStorage::new();

    // Pre-populate some data
    storage.put("existing", b"data").unwrap();

    {
        let mut tx = Transaction::new(&mut storage);

        // Add operations to transaction
        tx.put("key1", b"value1").unwrap();
        tx.put("key2", b"value2").unwrap();
        tx.delete("existing").unwrap();

        // Commit transaction
        tx.commit().unwrap();
    }

    // Verify changes were applied
    assert_eq!(storage.get("key1").unwrap(), Some(b"value1".to_vec()));
    assert_eq!(storage.get("key2").unwrap(), Some(b"value2".to_vec()));
    assert_eq!(storage.get("existing").unwrap(), None);
}

/// Test storage transaction rollback.
#[test]
fn test_storage_transaction_rollback() {
    let mut storage = MemoryStorage::new();

    // Pre-populate some data
    storage.put("existing", b"data").unwrap();

    {
        let mut tx = Transaction::new(&mut storage);

        // Add operations to transaction
        tx.put("key1", b"value1").unwrap();
        tx.delete("existing").unwrap();

        // Don't commit - transaction should rollback
    } // Transaction drops here without commit

    // Verify no changes were applied
    assert_eq!(storage.get("key1").unwrap(), None);
    assert_eq!(storage.get("existing").unwrap(), Some(b"data".to_vec()));
}

/// Test storage with unicode keys and values.
#[test]
fn test_unicode_storage() {
    let mut storage = MemoryStorage::new();

    let unicode_key = "🔑key_测试_مفتاح";
    let unicode_value = "🌟value_测试_قيمة";

    storage.put(unicode_key, unicode_value.as_bytes()).unwrap();

    let retrieved = storage.get(unicode_key).unwrap();
    assert_eq!(retrieved, Some(unicode_value.as_bytes().to_vec()));

    let keys = storage.list_keys("🔑").unwrap();
    assert_eq!(keys.len(), 1);
    assert_eq!(keys[0], unicode_key);
}

/// Test create_storage function with different backends.
#[test]
fn test_create_storage() {
    use zkp_dataset_ledger::storage::create_storage;

    // Test memory storage creation
    let storage = create_storage("memory", "").unwrap();

    // Verify it's working by doing a basic operation
    // Note: We can't test the actual operations because create_storage
    // returns a boxed trait object and we'd need mutable access
    assert!(storage.stats().is_ok());

    // Test invalid backend
    let result = create_storage("invalid_backend", "");
    assert!(result.is_err());
    if let Err(LedgerError::StorageError(msg)) = result {
        assert!(msg.contains("Unknown storage backend"));
    } else {
        panic!("Expected StorageError");
    }
}
