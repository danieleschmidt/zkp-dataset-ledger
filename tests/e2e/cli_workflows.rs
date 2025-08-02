use std::process::Command;
use std::fs;
use tempfile::TempDir;
use serde_json::Value;

/// End-to-end tests that exercise the complete CLI workflow
/// These tests verify that the CLI tool works correctly from a user perspective

pub struct CliTestRunner {
    temp_dir: TempDir,
    cli_binary_path: String,
}

impl CliTestRunner {
    pub fn new() -> Self {
        let temp_dir = TempDir::new().expect("Failed to create temp directory");
        
        // In a real test environment, this would point to the built binary
        let cli_binary_path = "target/debug/zkp-ledger".to_string();
        
        Self {
            temp_dir,
            cli_binary_path,
        }
    }
    
    pub fn temp_path(&self) -> &std::path::Path {
        self.temp_dir.path()
    }
    
    /// Run a CLI command and return the output
    pub fn run_command(&self, args: &[&str]) -> Result<std::process::Output, std::io::Error> {
        Command::new(&self.cli_binary_path)
            .args(args)
            .current_dir(self.temp_path())
            .output()
    }
    
    /// Run a CLI command and expect it to succeed
    pub fn run_command_success(&self, args: &[&str]) -> String {
        let output = self.run_command(args).expect("Failed to run command");
        
        if !output.status.success() {
            panic!(
                "Command failed: {} {}\nStdout: {}\nStderr: {}", 
                self.cli_binary_path,
                args.join(" "),
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
        }
        
        String::from_utf8_lossy(&output.stdout).to_string()
    }
    
    /// Run a CLI command and expect it to fail
    pub fn run_command_failure(&self, args: &[&str]) -> String {
        let output = self.run_command(args).expect("Failed to run command");
        
        if output.status.success() {
            panic!(
                "Command unexpectedly succeeded: {} {}\nStdout: {}", 
                self.cli_binary_path,
                args.join(" "),
                String::from_utf8_lossy(&output.stdout)
            );
        }
        
        String::from_utf8_lossy(&output.stderr).to_string()
    }
    
    /// Create a test CSV file
    pub fn create_test_csv(&self, name: &str, content: &str) -> std::path::PathBuf {
        let csv_path = self.temp_path().join(format!("{}.csv", name));
        fs::write(&csv_path, content).expect("Failed to write CSV file");
        csv_path
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_cli_help() {
        let runner = CliTestRunner::new();
        let output = runner.run_command_success(&["--help"]);
        
        assert!(output.contains("zkp-ledger"));
        assert!(output.contains("USAGE"));
        assert!(output.contains("SUBCOMMANDS"));
    }
    
    #[test]
    fn test_complete_workflow() {
        let runner = CliTestRunner::new();
        
        // Step 1: Initialize a new ledger
        let output = runner.run_command_success(&[
            "init", 
            "--project", "test-project",
            "--storage", "rocksdb"
        ]);
        assert!(output.contains("Initialized") || output.contains("Created"));
        
        // Step 2: Create test data
        let csv_content = r#"id,name,age,salary
1,Alice,30,75000
2,Bob,25,65000
3,Charlie,35,85000
4,Diana,28,70000
5,Eve,32,80000
"#;
        let csv_path = runner.create_test_csv("employees", csv_content);
        
        // Step 3: Notarize the dataset
        let output = runner.run_command_success(&[
            "notarize",
            csv_path.to_str().unwrap(),
            "--name", "employees-v1",
            "--output", "json"
        ]);
        
        assert!(output.contains("proof") || output.contains("transaction"));
        
        // Verify we can parse the output as JSON
        let json_output: Value = serde_json::from_str(&output)
            .expect("Output should be valid JSON");
        assert!(json_output.get("proof").is_some() || json_output.get("transaction_id").is_some());
        
        // Step 4: Verify the proof
        // Extract proof/transaction ID from previous output
        let proof_id = if let Some(proof) = json_output.get("proof") {
            proof.get("id").and_then(|v| v.as_str()).unwrap_or("latest")
        } else {
            "latest"
        };
        
        let output = runner.run_command_success(&[
            "verify",
            proof_id
        ]);
        assert!(output.contains("valid") || output.contains("verified"));
        
        // Step 5: Query audit trail
        let output = runner.run_command_success(&[
            "audit",
            "--format", "json"
        ]);
        
        let audit_json: Value = serde_json::from_str(&output)
            .expect("Audit output should be valid JSON");
        assert!(audit_json.is_array() || audit_json.get("transactions").is_some());
        
        // Step 6: Transform the dataset
        let transformed_csv = r#"id,name,age,salary,performance_score
1,Alice,30,75000,8.5
2,Bob,25,65000,7.2
3,Charlie,35,85000,9.1
4,Diana,28,70000,8.0
5,Eve,32,80000,8.8
"#;
        let transformed_path = runner.create_test_csv("employees_transformed", transformed_csv);
        
        let output = runner.run_command_success(&[
            "transform",
            "--input", "employees-v1",
            "--output-dataset", transformed_path.to_str().unwrap(),
            "--output-name", "employees-v2",
            "--operations", "add_column:performance_score"
        ]);
        
        assert!(output.contains("transformation") || output.contains("recorded"));
        
        // Step 7: Generate final audit report
        let output = runner.run_command_success(&[
            "report",
            "--from", "employees-v1",
            "--to", "employees-v2",
            "--format", "json-ld",
            "--include-proofs"
        ]);
        
        let report: Value = serde_json::from_str(&output)
            .expect("Report should be valid JSON-LD");
        assert!(report.get("@context").is_some());
        assert!(report.get("datasets").is_some() || report.get("provenance").is_some());
    }
    
    #[test]
    fn test_error_handling() {
        let runner = CliTestRunner::new();
        
        // Test with non-existent file
        let stderr = runner.run_command_failure(&[
            "notarize",
            "non_existent_file.csv",
            "--name", "test"
        ]);
        assert!(stderr.contains("not found") || stderr.contains("No such file"));
        
        // Test with malformed CSV
        let malformed_csv = runner.create_test_csv("malformed", "id,name\n1,Alice\n2,Bob,extra");
        let stderr = runner.run_command_failure(&[
            "notarize",
            malformed_csv.to_str().unwrap(),
            "--name", "malformed-test"
        ]);
        assert!(stderr.contains("parse") || stderr.contains("malformed") || stderr.contains("CSV"));
        
        // Test verify with non-existent proof
        let stderr = runner.run_command_failure(&[
            "verify",
            "non-existent-proof-id"
        ]);
        assert!(stderr.contains("not found") || stderr.contains("does not exist"));
    }
    
    #[test]
    fn test_configuration_options() {
        let runner = CliTestRunner::new();
        
        // Test different storage backends
        for backend in &["rocksdb", "memory"] {
            let output = runner.run_command_success(&[
                "init",
                "--project", &format!("test-{}", backend),
                "--storage", backend
            ]);
            assert!(output.contains("Initialized") || output.contains("Created"));
        }
        
        // Test different output formats
        let csv_content = "id,value\n1,100\n2,200\n";
        let csv_path = runner.create_test_csv("format_test", csv_content);
        
        for format in &["json", "yaml", "table"] {
            let output = runner.run_command_success(&[
                "notarize",
                csv_path.to_str().unwrap(),
                "--name", &format!("test-{}", format),
                "--output", format
            ]);
            
            match *format {
                "json" => {
                    serde_json::from_str::<Value>(&output)
                        .expect("JSON output should be valid");
                }
                "yaml" => {
                    assert!(output.contains(":") && (output.contains("-") || output.contains("proof")));
                }
                "table" => {
                    assert!(output.contains("|") || output.contains("─"));
                }
                _ => {}
            }
        }
    }
    
    #[test]
    fn test_batch_operations() {
        let runner = CliTestRunner::new();
        
        // Initialize ledger
        runner.run_command_success(&[
            "init", 
            "--project", "batch-test"
        ]);
        
        // Create multiple test files
        let datasets = vec![
            ("dataset1", "id,value\n1,100\n2,200\n"),
            ("dataset2", "id,score\n1,85\n2,92\n3,78\n"),
            ("dataset3", "name,age\nAlice,30\nBob,25\nCharlie,35\n"),
        ];
        
        let mut csv_paths = Vec::new();
        for (name, content) in datasets {
            let path = runner.create_test_csv(name, content);
            csv_paths.push(path);
        }
        
        // Batch notarize multiple datasets
        let mut args = vec!["batch-notarize", "--output", "json"];
        for (i, path) in csv_paths.iter().enumerate() {
            args.push("--dataset");
            args.push(path.to_str().unwrap());
            args.push("--name");
            args.push(&format!("batch-dataset-{}", i + 1));
        }
        
        let output = runner.run_command_success(&args);
        let batch_result: Value = serde_json::from_str(&output)
            .expect("Batch output should be valid JSON");
        
        // Should have results for all datasets
        if let Some(results) = batch_result.get("results").and_then(|v| v.as_array()) {
            assert_eq!(results.len(), 3);
        } else if let Some(transactions) = batch_result.get("transactions").and_then(|v| v.as_array()) {
            assert_eq!(transactions.len(), 3);
        } else {
            panic!("Batch result should contain results or transactions array");
        }
        
        // Verify all proofs
        let output = runner.run_command_success(&[
            "verify-all",
            "--format", "json"
        ]);
        
        let verify_result: Value = serde_json::from_str(&output)
            .expect("Verify all output should be valid JSON");
        assert!(verify_result.get("all_valid").is_some() || verify_result.get("results").is_some());
    }
    
    #[test]
    fn test_performance_commands() {
        let runner = CliTestRunner::new();
        
        // Initialize ledger
        runner.run_command_success(&[
            "init", 
            "--project", "perf-test"
        ]);
        
        // Create a larger dataset for performance testing
        let mut large_csv_content = String::from("id,name,value,category\n");
        for i in 1..=1000 {
            large_csv_content.push_str(&format!("{},item_{},{:.2},cat_{}\n", 
                i, i, i as f64 * 1.5, i % 10));
        }
        let large_csv = runner.create_test_csv("large_dataset", &large_csv_content);
        
        // Test benchmark command
        let output = runner.run_command_success(&[
            "benchmark",
            large_csv.to_str().unwrap(),
            "--iterations", "3",
            "--output", "json"
        ]);
        
        let benchmark_result: Value = serde_json::from_str(&output)
            .expect("Benchmark output should be valid JSON");
        
        assert!(benchmark_result.get("avg_proof_time").is_some() || 
                benchmark_result.get("performance").is_some());
        
        // Verify performance metrics are reasonable
        if let Some(avg_time) = benchmark_result.get("avg_proof_time").and_then(|v| v.as_f64()) {
            assert!(avg_time > 0.0, "Average proof time should be positive");
            assert!(avg_time < 60.0, "Average proof time should be reasonable for test data");
        }
    }
    
    #[test]
    fn test_export_functionality() {
        let runner = CliTestRunner::new();
        
        // Setup ledger with some data
        runner.run_command_success(&["init", "--project", "export-test"]);
        
        let csv_content = "id,product,price\n1,Widget,19.99\n2,Gadget,29.99\n3,Tool,39.99\n";
        let csv_path = runner.create_test_csv("products", csv_content);
        
        runner.run_command_success(&[
            "notarize",
            csv_path.to_str().unwrap(),
            "--name", "products-v1"
        ]);
        
        // Test different export formats
        let export_formats = vec![
            ("json-ld", "model-card.jsonld"),
            ("html", "report.html"),
            ("pdf", "audit-report.pdf"),
        ];
        
        for (format, filename) in export_formats {
            let output_path = runner.temp_path().join(filename);
            
            let _output = runner.run_command_success(&[
                "export",
                "--dataset", "products-v1",
                "--format", format,
                "--output", output_path.to_str().unwrap(),
                "--include-proofs",
                "--include-metadata"
            ]);
            
            // Verify file was created
            assert!(output_path.exists(), "Export file should be created: {}", filename);
            
            // Basic content validation
            let content = fs::read_to_string(&output_path).unwrap();
            match format {
                "json-ld" => {
                    let json: Value = serde_json::from_str(&content)
                        .expect("JSON-LD should be valid JSON");
                    assert!(json.get("@context").is_some());
                }
                "html" => {
                    assert!(content.contains("<html>") || content.contains("<!DOCTYPE"));
                    assert!(content.contains("products-v1"));
                }
                "pdf" => {
                    // PDF files start with %PDF
                    assert!(content.starts_with("%PDF"));
                }
                _ => {}
            }
        }
    }
}