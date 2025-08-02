use std::path::{Path, PathBuf};
use std::fs;
use tempfile::TempDir;
use serde_json::Value;

/// Test fixture utilities for ZKP Dataset Ledger testing
pub struct TestFixtures {
    pub temp_dir: TempDir,
}

impl TestFixtures {
    pub fn new() -> Self {
        Self {
            temp_dir: TempDir::new().expect("Failed to create temp directory")
        }
    }

    /// Get the path to the temporary directory
    pub fn temp_path(&self) -> &Path {
        self.temp_dir.path()
    }

    /// Create a simple CSV dataset for testing
    pub fn create_simple_csv(&self, name: &str, rows: usize) -> PathBuf {
        let csv_path = self.temp_path().join(format!("{}.csv", name));
        let mut content = String::from("id,name,value,category\n");
        
        for i in 1..=rows {
            content.push_str(&format!("{},item_{},{:.2},category_{}\n", 
                i, i, i as f64 * 10.5, i % 3));
        }
        
        fs::write(&csv_path, content).expect("Failed to write CSV file");
        csv_path
    }

    /// Create a CSV with statistical properties for testing
    pub fn create_statistical_csv(&self, name: &str) -> PathBuf {
        let csv_path = self.temp_path().join(format!("{}.csv", name));
        let content = r#"id,age,salary,department,performance_score
1,25,50000.00,Engineering,8.5
2,30,75000.00,Engineering,9.2
3,28,65000.00,Marketing,7.8
4,35,85000.00,Engineering,9.0
5,22,45000.00,Marketing,6.5
6,40,95000.00,Management,9.5
7,27,60000.00,Marketing,8.0
8,33,80000.00,Engineering,8.8
9,29,70000.00,Engineering,8.3
10,45,120000.00,Management,9.8
"#;
        
        fs::write(&csv_path, content).expect("Failed to write statistical CSV");
        csv_path
    }

    /// Create a large CSV dataset for performance testing
    pub fn create_large_csv(&self, name: &str, rows: usize) -> PathBuf {
        let csv_path = self.temp_path().join(format!("{}.csv", name));
        let mut content = String::from("id,timestamp,user_id,action,value,metadata\n");
        
        for i in 1..=rows {
            content.push_str(&format!("{},{},{},action_{},{:.4},metadata_{}\n",
                i,
                1640995200 + (i * 60), // Incremental timestamps
                i % 1000,
                i % 10,
                i as f64 * 0.123,
                i
            ));
        }
        
        fs::write(&csv_path, content).expect("Failed to write large CSV");
        csv_path
    }

    /// Create a JSON dataset for testing
    pub fn create_json_dataset(&self, name: &str) -> PathBuf {
        let json_path = self.temp_path().join(format!("{}.json", name));
        let data = serde_json::json!([
            {
                "id": 1,
                "user": {
                    "name": "Alice",
                    "age": 30,
                    "email": "alice@example.com"
                },
                "transactions": [
                    {"amount": 100.50, "type": "credit"},
                    {"amount": 25.00, "type": "debit"}
                ]
            },
            {
                "id": 2,
                "user": {
                    "name": "Bob",
                    "age": 25,
                    "email": "bob@example.com"
                },
                "transactions": [
                    {"amount": 200.00, "type": "credit"}
                ]
            }
        ]);
        
        fs::write(&json_path, serde_json::to_string_pretty(&data).unwrap())
            .expect("Failed to write JSON file");
        json_path
    }

    /// Create test circuit input data
    pub fn create_circuit_input(&self, name: &str) -> PathBuf {
        let input_path = self.temp_path().join(format!("{}_circuit.json", name));
        let circuit_input = serde_json::json!({
            "public_inputs": {
                "dataset_hash": "0x1234567890abcdef",
                "row_count": 1000,
                "column_count": 5
            },
            "private_inputs": {
                "row_data": [
                    [1, "Alice", 30, "Engineering", 8.5],
                    [2, "Bob", 25, "Marketing", 7.2]
                ],
                "statistical_properties": {
                    "mean_age": 27.5,
                    "salary_variance": 125000.0
                }
            },
            "constraints": {
                "min_rows": 100,
                "max_age": 65,
                "required_columns": ["id", "name", "age"]
            }
        });
        
        fs::write(&input_path, serde_json::to_string_pretty(&circuit_input).unwrap())
            .expect("Failed to write circuit input");
        input_path
    }

    /// Create malformed CSV for error testing
    pub fn create_malformed_csv(&self, name: &str) -> PathBuf {
        let csv_path = self.temp_path().join(format!("{}.csv", name));
        let content = r#"id,name,value
1,Alice,100
2,Bob
3,Charlie,300,extra_column
4,"Unclosed quote,400
"#;
        
        fs::write(&csv_path, content).expect("Failed to write malformed CSV");
        csv_path
    }

    /// Create empty dataset for edge case testing
    pub fn create_empty_csv(&self, name: &str) -> PathBuf {
        let csv_path = self.temp_path().join(format!("{}.csv", name));
        fs::write(&csv_path, "").expect("Failed to write empty CSV");
        csv_path
    }

    /// Create CSV with only headers
    pub fn create_headers_only_csv(&self, name: &str) -> PathBuf {
        let csv_path = self.temp_path().join(format!("{}.csv", name));
        fs::write(&csv_path, "id,name,value\n").expect("Failed to write headers-only CSV");
        csv_path
    }

    /// Create test configuration file
    pub fn create_test_config(&self, storage_backend: &str) -> PathBuf {
        let config_path = self.temp_path().join("test_config.toml");
        let config_content = format!(r#"
[ledger]
name = "test-ledger"
hash_algorithm = "sha3-256"

[storage]
backend = "{}"
path = "{}/test_ledger"

[proof]
curve = "bls12-381"
security_level = 128
parallel_prove = false  # Disable for deterministic testing

[testing]
deterministic_rng = true
skip_heavy_operations = false
"#, storage_backend, self.temp_path().display());
        
        fs::write(&config_path, config_content).expect("Failed to write config file");
        config_path
    }

    /// Generate test trusted setup parameters
    pub fn create_test_trusted_setup(&self) -> PathBuf {
        let setup_dir = self.temp_path().join("trusted_setup");
        fs::create_dir_all(&setup_dir).expect("Failed to create trusted setup dir");
        
        // Create placeholder files (in real implementation, these would be actual parameters)
        let proving_key = setup_dir.join("proving.key");
        let verifying_key = setup_dir.join("verifying.key");
        
        fs::write(&proving_key, b"mock_proving_key_data").expect("Failed to write proving key");
        fs::write(&verifying_key, b"mock_verifying_key_data").expect("Failed to write verifying key");
        
        setup_dir
    }
}

impl Default for TestFixtures {
    fn default() -> Self {
        Self::new()
    }
}

/// Common test utilities
pub mod utils {
    use super::*;
    
    /// Assert that two f64 values are approximately equal
    pub fn assert_approx_eq(a: f64, b: f64, epsilon: f64) {
        assert!((a - b).abs() < epsilon, "Values {} and {} are not approximately equal", a, b);
    }
    
    /// Generate deterministic test data
    pub fn generate_test_seed() -> u64 {
        42 // Fixed seed for reproducible tests
    }
    
    /// Create a mock proof for testing
    pub fn create_mock_proof() -> serde_json::Value {
        serde_json::json!({
            "proof_data": "0x1234567890abcdef",
            "public_inputs": ["0xabcdef", "1000", "5"],
            "verification_key": "0xfedcba0987654321",
            "timestamp": 1640995200,
            "dataset_hash": "0x1111222233334444"
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_fixtures_creation() {
        let fixtures = TestFixtures::new();
        assert!(fixtures.temp_path().exists());
        
        let csv_path = fixtures.create_simple_csv("test", 10);
        assert!(csv_path.exists());
        
        let content = fs::read_to_string(csv_path).unwrap();
        assert!(content.contains("id,name,value,category"));
        assert!(content.lines().count() == 11); // Header + 10 rows
    }
    
    #[test]
    fn test_large_dataset_creation() {
        let fixtures = TestFixtures::new();
        let csv_path = fixtures.create_large_csv("large", 1000);
        
        let content = fs::read_to_string(csv_path).unwrap();
        assert!(content.lines().count() == 1001); // Header + 1000 rows
    }
    
    #[test]
    fn test_malformed_data_creation() {
        let fixtures = TestFixtures::new();
        let csv_path = fixtures.create_malformed_csv("bad");
        
        let content = fs::read_to_string(csv_path).unwrap();
        assert!(content.contains("Unclosed quote"));
    }
}