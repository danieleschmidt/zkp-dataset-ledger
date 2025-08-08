//! Unit tests for dataset functionality.

use std::io::Write;
use tempfile::NamedTempFile;
use zkp_dataset_ledger::{Dataset, Result};

/// Test basic dataset creation from CSV.
#[test]
fn test_dataset_basic_creation() -> Result<()> {
    let mut temp_file = NamedTempFile::new().unwrap();
    writeln!(temp_file, "id,value")?;
    writeln!(temp_file, "1,100")?;
    writeln!(temp_file, "2,200")?;

    let temp_path = temp_file.path().with_extension("csv");
    std::fs::copy(temp_file.path(), &temp_path)?;

    let dataset = Dataset::from_path(&temp_path)?;

    assert_eq!(dataset.row_count, Some(2));
    assert_eq!(dataset.column_count, Some(2));
    assert!(dataset.size > 0);

    std::fs::remove_file(temp_path).ok();
    Ok(())
}

/// Test dataset hash computation.
#[test]
fn test_dataset_hash() -> Result<()> {
    let mut temp_file = NamedTempFile::new().unwrap();
    writeln!(temp_file, "a,b")?;
    writeln!(temp_file, "1,2")?;

    let temp_path = temp_file.path().with_extension("csv");
    std::fs::copy(temp_file.path(), &temp_path)?;

    let dataset = Dataset::from_path(&temp_path)?;
    let hash = dataset.compute_hash();

    assert!(!hash.is_empty());
    assert_eq!(hash.len(), 64); // SHA-256 hex string length

    std::fs::remove_file(temp_path).ok();
    Ok(())
}

/// Test dataset from memory string.
#[test]
fn test_dataset_from_string() -> Result<()> {
    let csv_content = "name,age\nAlice,25\nBob,30";
    let dataset = Dataset::from_csv_string(csv_content)?;

    assert_eq!(dataset.row_count, Some(2));
    assert_eq!(dataset.column_count, Some(2));
    assert!(dataset.size > 0);
    assert!(dataset.path.is_none());

    Ok(())
}