// Test fixtures and utilities for ZKP Dataset Ledger
use serde_json::Value;
use std::path::{Path, PathBuf};
use tempfile::{NamedTempFile, TempDir};

/// Test dataset generator for creating sample data
pub struct TestDataGenerator {
    temp_dir: TempDir,
}

impl TestDataGenerator {
    pub fn new() -> Self {
        Self {
            temp_dir: TempDir::new().expect("Failed to create temp directory"),
        }
    }

    pub fn temp_path(&self) -> &Path {
        self.temp_dir.path()
    }

    /// Generate a small CSV dataset for quick testing
    pub fn create_small_csv(&self, name: &str) -> PathBuf {
        let path = self.temp_dir.path().join(format!("{}.csv", name));
        let content = "id,feature_1,feature_2,category,target\n\
                      1,0.5,-0.2,A,1\n\
                      2,-0.3,1.1,B,0\n\
                      3,0.8,0.4,A,1\n\
                      4,-1.2,-0.7,C,0\n\
                      5,0.1,0.9,B,1\n";
        std::fs::write(&path, content).expect("Failed to write test CSV");
        path
    }

    /// Generate a medium CSV dataset for performance testing
    pub fn create_medium_csv(&self, name: &str, rows: usize) -> PathBuf {
        let path = self.temp_dir.path().join(format!("{}.csv", name));
        let mut content = String::from("id,feature_1,feature_2,category,target\n");

        for i in 1..=rows {
            content.push_str(&format!(
                "{},{:.3},{:.3},{},{}\n",
                i,
                (i as f64 * 0.1).sin(),
                (i as f64 * 0.2).cos(),
                match i % 3 {
                    0 => "A",
                    1 => "B",
                    _ => "C",
                },
                i % 2
            ));
        }

        std::fs::write(&path, content).expect("Failed to write test CSV");
        path
    }

    /// Generate a JSON dataset
    pub fn create_json_dataset(&self, name: &str) -> PathBuf {
        let path = self.temp_dir.path().join(format!("{}.json", name));
        let data = serde_json::json!([
            {"id": 1, "feature_1": 0.5, "feature_2": -0.2, "category": "A", "target": 1},
            {"id": 2, "feature_1": -0.3, "feature_2": 1.1, "category": "B", "target": 0},
            {"id": 3, "feature_1": 0.8, "feature_2": 0.4, "category": "A", "target": 1},
        ]);
        std::fs::write(&path, serde_json::to_string_pretty(&data).unwrap())
            .expect("Failed to write test JSON");
        path
    }

    /// Generate a dataset with missing values
    pub fn create_dataset_with_nulls(&self, name: &str) -> PathBuf {
        let path = self.temp_dir.path().join(format!("{}.csv", name));
        let content = "id,feature_1,feature_2,category,target\n\
                      1,0.5,,A,1\n\
                      2,,1.1,B,0\n\
                      3,0.8,0.4,,1\n\
                      4,-1.2,-0.7,C,\n\
                      5,0.1,0.9,B,1\n";
        std::fs::write(&path, content).expect("Failed to write test CSV");
        path
    }

    /// Generate a dataset with specific schema for testing
    pub fn create_typed_dataset(&self, name: &str) -> PathBuf {
        let path = self.temp_dir.path().join(format!("{}.csv", name));
        let content = "user_id,timestamp,age,balance,active,email\n\
                      1001,2024-01-15T10:30:00Z,25,1250.50,true,user1@example.com\n\
                      1002,2024-01-15T11:45:00Z,34,2750.25,false,user2@example.com\n\
                      1003,2024-01-15T12:15:00Z,28,850.75,true,user3@example.com\n\
                      1004,2024-01-15T13:20:00Z,42,3200.00,true,user4@example.com\n\
                      1005,2024-01-15T14:30:00Z,31,1850.25,false,user5@example.com\n";
        std::fs::write(&path, content).expect("Failed to write test CSV");
        path
    }

    /// Generate a malformed dataset for error testing
    pub fn create_malformed_csv(&self, name: &str) -> PathBuf {
        let path = self.temp_dir.path().join(format!("{}.csv", name));
        let content = "id,feature_1,feature_2\n\
                      1,0.5,-0.2,extra_column\n\
                      2,-0.3\n\
                      invalid_row\n\
                      4,-1.2,-0.7\n";
        std::fs::write(&path, content).expect("Failed to write test CSV");
        path
    }

    /// Generate binary data file for edge case testing
    pub fn create_binary_file(&self, name: &str) -> PathBuf {
        let path = self.temp_dir.path().join(format!("{}.bin", name));
        let data: Vec<u8> = (0..=255).collect();
        std::fs::write(&path, data).expect("Failed to write binary file");
        path
    }
}

/// Mock proof configuration for testing
pub struct MockProofConfig {
    pub generate_actual_proof: bool,
    pub simulate_delay_ms: u64,
    pub force_verification_result: Option<bool>,
}

impl Default for MockProofConfig {
    fn default() -> Self {
        Self {
            generate_actual_proof: false,
            simulate_delay_ms: 0,
            force_verification_result: Some(true),
        }
    }
}

impl MockProofConfig {
    pub fn real_proof() -> Self {
        Self {
            generate_actual_proof: true,
            simulate_delay_ms: 0,
            force_verification_result: None,
        }
    }

    pub fn with_delay(delay_ms: u64) -> Self {
        Self {
            generate_actual_proof: false,
            simulate_delay_ms: delay_ms,
            force_verification_result: Some(true),
        }
    }

    pub fn failing_verification() -> Self {
        Self {
            generate_actual_proof: false,
            simulate_delay_ms: 0,
            force_verification_result: Some(false),
        }
    }
}

/// Test helper for creating ledger instances
pub struct TestLedger {
    pub temp_dir: TempDir,
    pub ledger_path: PathBuf,
}

impl TestLedger {
    pub fn new() -> Self {
        let temp_dir = TempDir::new().expect("Failed to create temp directory");
        let ledger_path = temp_dir.path().join("test_ledger");
        Self {
            temp_dir,
            ledger_path,
        }
    }

    pub fn with_name(name: &str) -> Self {
        let temp_dir = TempDir::new().expect("Failed to create temp directory");
        let ledger_path = temp_dir.path().join(name);
        Self {
            temp_dir,
            ledger_path,
        }
    }

    pub fn path(&self) -> &Path {
        &self.ledger_path
    }

    pub fn temp_path(&self) -> &Path {
        self.temp_dir.path()
    }
}

/// Performance testing utilities
pub struct PerformanceTester {
    pub start_time: std::time::Instant,
}

impl PerformanceTester {
    pub fn new() -> Self {
        Self {
            start_time: std::time::Instant::now(),
        }
    }

    pub fn elapsed_ms(&self) -> u128 {
        self.start_time.elapsed().as_millis()
    }

    pub fn elapsed_secs(&self) -> f64 {
        self.start_time.elapsed().as_secs_f64()
    }

    pub fn assert_under_ms(&self, max_ms: u128, operation: &str) {
        let elapsed = self.elapsed_ms();
        assert!(
            elapsed <= max_ms,
            "{} took {}ms, expected under {}ms",
            operation,
            elapsed,
            max_ms
        );
    }

    pub fn assert_under_secs(&self, max_secs: f64, operation: &str) {
        let elapsed = self.elapsed_secs();
        assert!(
            elapsed <= max_secs,
            "{} took {:.2}s, expected under {:.2}s",
            operation,
            elapsed,
            max_secs
        );
    }
}

/// Test constants and configuration
pub mod constants {
    pub const SMALL_DATASET_ROWS: usize = 100;
    pub const MEDIUM_DATASET_ROWS: usize = 10_000;
    pub const LARGE_DATASET_ROWS: usize = 100_000;

    pub const MAX_PROOF_TIME_MS: u128 = 5_000;
    pub const MAX_VERIFICATION_TIME_MS: u128 = 100;
    pub const MAX_LEDGER_INIT_TIME_MS: u128 = 1_000;

    pub const TEST_PROOF_TIMEOUT_SECS: u64 = 30;
    pub const TEST_LEDGER_MAX_SIZE_MB: usize = 100;
}

/// Assertion helpers for cryptographic properties
pub mod assertions {
    use super::*;

    pub fn assert_proof_valid(proof: &zkp_dataset_ledger::Proof) {
        assert!(!proof.is_empty(), "Proof should not be empty");
        // Add more cryptographic property assertions as needed
    }

    pub fn assert_dataset_properties(
        dataset: &zkp_dataset_ledger::Dataset,
        expected_rows: usize,
        expected_cols: usize,
    ) {
        assert_eq!(dataset.row_count(), expected_rows, "Row count mismatch");
        assert_eq!(
            dataset.column_count(),
            expected_cols,
            "Column count mismatch"
        );
    }

    pub fn assert_ledger_integrity(ledger: &zkp_dataset_ledger::Ledger) {
        assert!(ledger
            .verify_chain_integrity()
            .expect("Chain integrity check failed"));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_data_generator_creates_valid_csv() {
        let generator = TestDataGenerator::new();
        let csv_path = generator.create_small_csv("test");

        assert!(csv_path.exists());
        let content = std::fs::read_to_string(&csv_path).unwrap();
        assert!(content.contains("id,feature_1"));
        assert!(content.lines().count() > 1); // Header + data rows
    }

    #[test]
    fn test_performance_tester() {
        let tester = PerformanceTester::new();
        std::thread::sleep(std::time::Duration::from_millis(10));
        assert!(tester.elapsed_ms() >= 10);
    }

    #[test]
    fn test_ledger_helper() {
        let test_ledger = TestLedger::new();
        assert!(test_ledger.temp_path().exists());
        assert!(!test_ledger.path().exists()); // Ledger not created yet
    }
}
