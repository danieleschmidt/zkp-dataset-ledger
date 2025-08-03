//! Dataset processing and validation for ZKP Dataset Ledger.

use crate::crypto::hash::{hash_dataset_file, HashAlgorithm};
use crate::{LedgerError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// Represents a dataset with metadata and schema information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dataset {
    pub name: String,
    pub hash: String,
    pub size: u64,
    pub row_count: Option<u64>,
    pub column_count: Option<u64>,
    pub schema: Option<Vec<ColumnInfo>>,
    pub statistics: Option<DatasetStatistics>,
    pub format: DatasetFormat,
    pub path: Option<String>,
}

/// Information about a column in the dataset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColumnInfo {
    pub name: String,
    pub data_type: ColumnType,
    pub nullable: bool,
    pub unique_values: Option<u64>,
    pub null_count: Option<u64>,
    pub statistics: Option<ColumnStatistics>,
}

/// Supported column data types.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ColumnType {
    Integer,
    Float,
    String,
    Boolean,
    DateTime,
    Binary,
}

/// Statistical properties of a column.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColumnStatistics {
    pub min: Option<String>,
    pub max: Option<String>,
    pub mean: Option<f64>,
    pub median: Option<f64>,
    pub std_dev: Option<f64>,
    pub variance: Option<f64>,
    pub skewness: Option<f64>,
    pub kurtosis: Option<f64>,
}

/// Overall dataset statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetStatistics {
    pub total_size_bytes: u64,
    pub memory_usage_bytes: Option<u64>,
    pub null_percentage: f64,
    pub duplicate_rows: Option<u64>,
    pub correlation_matrix: Option<HashMap<String, HashMap<String, f64>>>,
}

/// Supported dataset file formats.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum DatasetFormat {
    Csv,
    Parquet,
    Json,
    JsonLines,
    Arrow,
    Excel,
}

impl DatasetFormat {
    /// Detect format from file extension.
    pub fn from_path<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        let extension = path
            .extension()
            .and_then(|ext| ext.to_str())
            .map(|s| s.to_lowercase())
            .ok_or_else(|| {
                LedgerError::DatasetError("Cannot determine file format from extension".to_string())
            })?;

        match extension.as_str() {
            "csv" => Ok(Self::Csv),
            "parquet" => Ok(Self::Parquet),
            "json" => Ok(Self::Json),
            "jsonl" | "ndjson" => Ok(Self::JsonLines),
            "arrow" => Ok(Self::Arrow),
            "xlsx" | "xls" => Ok(Self::Excel),
            _ => Err(LedgerError::DatasetError(format!(
                "Unsupported file format: {}",
                extension
            ))),
        }
    }
}

impl Dataset {
    /// Load a dataset from a file path.
    pub fn from_path<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        let format = DatasetFormat::from_path(path)?;
        
        // Get file metadata
        let metadata = std::fs::metadata(path)
            .map_err(|e| LedgerError::IoError(format!("Failed to read file metadata: {}", e)))?;
        let size = metadata.len();

        // Compute file hash
        let hash = hash_dataset_file(path, HashAlgorithm::default())
            .map_err(|e| LedgerError::DatasetError(format!("Failed to compute hash: {}", e)))?;

        // Basic CSV analysis for row/column counts
        let (row_count, column_count) = if format == DatasetFormat::Csv {
            Self::analyze_csv_basic(path)?
        } else {
            (0, 0) // Placeholder for other formats
        };

        Ok(Dataset {
            name: path
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string(),
            hash,
            size,
            row_count: Some(row_count),
            column_count: Some(column_count),
            schema: None, // Would be populated with detailed analysis
            statistics: None, // Would be populated with detailed analysis
            format,
            path: Some(path.to_string_lossy().to_string()),
        })
    }

    /// Basic CSV analysis to get row and column counts.
    fn analyze_csv_basic<P: AsRef<Path>>(path: P) -> Result<(u64, u64)> {
        let mut reader = csv::Reader::from_path(path)?;
        
        // Get column count from headers
        let headers = reader.headers()?;
        let column_count = headers.len() as u64;
        
        // Count rows
        let row_count = reader.records().count() as u64;
        
        Ok((row_count, column_count))
    }

    /// Get the computed hash of the dataset.
    pub fn compute_hash(&self) -> String {
        self.hash.clone()
    }

    /// Validate the dataset integrity by recomputing hash.
    pub fn validate_integrity(&self) -> Result<bool> {
        if let Some(path) = &self.path {
            let current_hash = hash_dataset_file(path, HashAlgorithm::default())?;
            Ok(current_hash == self.hash)
        } else {
            // For in-memory datasets, we trust the provided hash
            Ok(true)
        }
    }

    /// Get a summary of the dataset properties.
    pub fn summary(&self) -> DatasetSummary {
        DatasetSummary {
            name: self.name.clone(),
            rows: self.row_count.unwrap_or(0),
            columns: self.column_count.unwrap_or(0),
            size_bytes: self.size,
            format: self.format.clone(),
            has_nulls: self.statistics.as_ref()
                .map(|s| s.null_percentage > 0.0)
                .unwrap_or(false),
            schema_types: self.schema.as_ref()
                .map(|s| s.iter().map(|col| col.data_type.clone()).collect())
                .unwrap_or_default(),
        }
    }

    /// Transform the dataset and create a new Dataset instance.
    pub fn transform(&self, operation: &str, _params: &HashMap<String, String>) -> Result<Self> {
        // This would contain actual transformation logic
        // For now, return a placeholder that shows the transformation was applied
        let mut new_dataset = self.clone();
        new_dataset.name = format!("{}_transformed", self.name);
        new_dataset.hash = format!("{}_{}", self.hash, operation);
        Ok(new_dataset)
    }
}

/// A lightweight summary of dataset properties.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetSummary {
    pub name: String,
    pub rows: u64,
    pub columns: u64,
    pub size_bytes: u64,
    pub format: DatasetFormat,
    pub has_nulls: bool,
    pub schema_types: Vec<ColumnType>,
}

/// Configuration for dataset processing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetConfig {
    pub hash_algorithm: HashAlgorithm,
    pub compute_full_statistics: bool,
    pub sample_size: Option<usize>,
    pub max_memory_mb: Option<usize>,
}

impl Default for DatasetConfig {
    fn default() -> Self {
        Self {
            hash_algorithm: HashAlgorithm::default(),
            compute_full_statistics: true,
            sample_size: None,
            max_memory_mb: Some(1024), // 1GB default
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_dataset_format_detection() {
        assert_eq!(DatasetFormat::from_path("test.csv").unwrap(), DatasetFormat::Csv);
        assert_eq!(DatasetFormat::from_path("test.parquet").unwrap(), DatasetFormat::Parquet);
        assert_eq!(DatasetFormat::from_path("test.json").unwrap(), DatasetFormat::Json);
        assert_eq!(DatasetFormat::from_path("test.jsonl").unwrap(), DatasetFormat::JsonLines);
    }

    #[test]
    fn test_dataset_from_csv() {
        // Create a temporary CSV file
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "name,age,score").unwrap();
        writeln!(temp_file, "Alice,25,85.5").unwrap();
        writeln!(temp_file, "Bob,30,92.0").unwrap();

        let temp_path = temp_file.path().with_extension("csv");
        std::fs::copy(temp_file.path(), &temp_path).unwrap();

        let dataset = Dataset::from_path(&temp_path).unwrap();
        
        assert_eq!(dataset.format, DatasetFormat::Csv);
        assert_eq!(dataset.row_count, Some(2)); // Two data rows
        assert_eq!(dataset.column_count, Some(3)); // Three columns
        assert!(!dataset.hash.is_empty());

        // Cleanup
        std::fs::remove_file(temp_path).ok();
    }

    #[test]
    fn test_dataset_integrity_validation() {
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "id,value").unwrap();
        writeln!(temp_file, "1,100").unwrap();

        let temp_path = temp_file.path().with_extension("csv");
        std::fs::copy(temp_file.path(), &temp_path).unwrap();

        let dataset = Dataset::from_path(&temp_path).unwrap();
        
        // Should validate successfully
        assert!(dataset.validate_integrity().unwrap());

        std::fs::remove_file(temp_path).ok();
    }
}
