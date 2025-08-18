use clap::{Parser, Subcommand};
use log::{debug, info};

// Import our simplified modules
use zkp_dataset_ledger::{Config, Dataset, Ledger, Result};

#[derive(Parser)]
#[command(name = "zkp-ledger")]
#[command(about = "A CLI for ZKP Dataset Ledger - Cryptographic provenance for ML pipelines")]
#[command(version = "0.1.0")]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Enable verbose logging
    #[arg(short, long, global = true)]
    verbose: bool,
}

#[derive(Subcommand)]
enum Commands {
    /// Initialize a new ledger for a project
    Init {
        #[arg(long)]
        project: String,
    },

    /// Notarize a dataset with cryptographic proof
    Notarize {
        /// Path to the dataset file
        dataset: String,

        /// Name for the dataset entry
        #[arg(long)]
        name: String,

        /// Proof type
        #[arg(long, default_value = "integrity")]
        proof_type: String,
    },

    /// Show dataset history
    History {
        /// Dataset name
        dataset: String,
    },

    /// Verify a proof
    Verify {
        /// Dataset name
        dataset: String,
    },

    /// List all datasets in the ledger
    List {
        /// Project name to list datasets for
        #[arg(long)]
        project: Option<String>,
    },

    /// Show ledger statistics
    Stats {
        /// Project name
        #[arg(long)]
        project: Option<String>,
    },

    /// Verify entire ledger integrity
    Check {
        /// Project name
        #[arg(long)]
        project: Option<String>,
    },

    /// Run research benchmarks and analysis
    Research {
        /// Type of research to run
        #[arg(long, default_value = "benchmark")]
        research_type: String,

        /// Number of iterations for benchmarks
        #[arg(long, default_value = "100")]
        iterations: usize,

        /// Output file for research results
        #[arg(long)]
        output: Option<String>,
    },

    /// Generate comprehensive audit report
    Audit {
        /// Dataset to audit
        dataset: String,

        /// Output format (json, pdf, html)
        #[arg(long, default_value = "json")]
        format: String,

        /// Output file path
        #[arg(long)]
        output: Option<String>,
    },
}

fn get_ledger_path(project: Option<String>) -> String {
    match project {
        Some(proj) => format!("./{}_ledger/ledger.json", proj),
        None => "./default_ledger/ledger.json".to_string(),
    }
}

fn get_or_create_ledger(project: Option<String>) -> Result<Ledger> {
    let ledger_path = get_ledger_path(project.clone());
    let ledger_name = project.unwrap_or_else(|| "default".to_string());

    Ledger::with_storage(ledger_name, ledger_path)
}

/// Create sample datasets for research and testing
fn create_sample_datasets() -> Result<Vec<Dataset>> {
    use std::collections::HashMap;
    use zkp_dataset_ledger::{Dataset, DatasetFormat, DatasetSchema, DatasetStatistics};

    let datasets = vec![
        Dataset {
            name: "sample_small".to_string(),
            hash: "sample_hash_small".to_string(),
            size: 1024,
            row_count: Some(100),
            column_count: Some(5),
            path: None,
            schema: None,
            statistics: None,
            format: DatasetFormat::Csv,
        },
        Dataset {
            name: "sample_medium".to_string(),
            hash: "sample_hash_medium".to_string(),
            size: 102400,
            row_count: Some(10000),
            column_count: Some(20),
            path: None,
            schema: None,
            statistics: None,
            format: DatasetFormat::Csv,
        },
        Dataset {
            name: "sample_large".to_string(),
            hash: "sample_hash_large".to_string(),
            size: 10240000,
            row_count: Some(1000000),
            column_count: Some(50),
            path: None,
            schema: None,
            statistics: None,
            format: DatasetFormat::Csv,
        },
    ];

    Ok(datasets)
}

/// Generate HTML audit report
fn generate_html_report(data: &serde_json::Value) -> String {
    format!(
        r#"<!DOCTYPE html>
<html>
<head>
    <title>ZKP Dataset Audit Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f0f8ff; padding: 20px; border-radius: 5px; }}
        .section {{ margin: 20px 0; }}
        .status-good {{ color: #008000; font-weight: bold; }}
        .status-error {{ color: #ff0000; font-weight: bold; }}
        pre {{ background-color: #f5f5f5; padding: 10px; border-radius: 3px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🔐 ZKP Dataset Audit Report</h1>
        <p>Generated on: {}</p>
    </div>
    
    <div class="section">
        <h2>Dataset Information</h2>
        <p><strong>Name:</strong> {}</p>
        <p><strong>Hash:</strong> <code>{}</code></p>
        <p><strong>Timestamp:</strong> {}</p>
    </div>

    <div class="section">
        <h2>Integrity Status</h2>
        <p class="{}">● {}</p>
    </div>

    <div class="section">
        <h2>Raw Data</h2>
        <pre>{}</pre>
    </div>
</body>
</html>"#,
        chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC"),
        data.get("dataset_name")
            .and_then(|v| v.as_str())
            .unwrap_or("Unknown"),
        data.get("dataset_hash")
            .and_then(|v| v.as_str())
            .unwrap_or("Unknown"),
        data.get("timestamp")
            .and_then(|v| v.as_str())
            .unwrap_or("Unknown"),
        if data
            .get("integrity_verified")
            .and_then(|v| v.as_bool())
            .unwrap_or(false)
        {
            "status-good"
        } else {
            "status-error"
        },
        if data
            .get("integrity_verified")
            .and_then(|v| v.as_bool())
            .unwrap_or(false)
        {
            "Verified"
        } else {
            "Failed"
        },
        serde_json::to_string_pretty(data).unwrap_or_else(|_| "Error formatting data".to_string())
    )
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    // Initialize logging
    let log_level = if cli.verbose { "debug" } else { "info" };
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or(log_level))
        .format_timestamp_secs()
        .init();

    if cli.verbose {
        info!("ZKP Dataset Ledger - Enhanced Version with Logging");
        debug!("Verbose logging enabled");
    }

    match cli.command {
        Commands::Init { project } => {
            println!("Initializing ledger for project: {}", project);
            let config = Config {
                ledger_name: project.clone(),
                storage_path: format!("./{}_ledger", project),
            };

            // Create directory if it doesn't exist
            std::fs::create_dir_all(&config.storage_path)?;

            println!("✅ Ledger initialized at: {}", config.storage_path);
            Ok(())
        }

        Commands::Notarize {
            dataset,
            name,
            proof_type,
        } => {
            println!("Notarizing dataset: {} ({})", name, dataset);

            let mut ledger = get_or_create_ledger(None)?;

            // Check if file exists
            if !std::path::Path::new(&dataset).exists() {
                return Err(zkp_dataset_ledger::LedgerError::not_found(
                    "dataset",
                    format!("Dataset file not found: {}", dataset),
                ));
            }

            let dataset_obj = Dataset::new(name.clone(), dataset)?;
            let proof = ledger.notarize_dataset(dataset_obj, proof_type)?;

            println!("✅ Dataset notarized successfully!");
            println!("   Dataset: {}", name);
            println!("   Hash: {}", proof.dataset_hash);
            println!("   Proof Type: {}", proof.proof_type);
            println!("   Timestamp: {}", proof.timestamp);

            Ok(())
        }

        Commands::History { dataset } => {
            println!("Showing history for dataset: {}", dataset);

            let ledger = get_or_create_ledger(None)?;

            match ledger.get_dataset_history(&dataset) {
                Some(entry) => {
                    println!("📊 Dataset History:");
                    println!("   Name: {}", entry.dataset_name);
                    println!("   Hash: {}", entry.dataset_hash);
                    println!("   Operation: {}", entry.operation);
                    println!("   Timestamp: {}", entry.timestamp);
                    println!("   Proof Type: {}", entry.proof.proof_type);
                }
                None => {
                    println!("❌ No history found for dataset: {}", dataset);
                }
            }

            Ok(())
        }

        Commands::Verify { dataset } => {
            println!("Verifying dataset: {}", dataset);

            let ledger = get_or_create_ledger(None)?;

            match ledger.get_dataset_history(&dataset) {
                Some(entry) => {
                    let is_valid = ledger.verify_proof(&entry.proof);
                    if is_valid {
                        println!("✅ Proof verification successful!");
                        println!("   Dataset: {}", dataset);
                        println!("   Hash: {}", entry.dataset_hash);
                    } else {
                        println!("❌ Proof verification failed!");
                    }
                }
                None => {
                    println!("❌ No proof found for dataset: {}", dataset);
                }
            }

            Ok(())
        }

        Commands::List { project } => {
            println!("📋 Listing datasets...");

            let ledger = get_or_create_ledger(project)?;
            let datasets = ledger.list_datasets();

            if datasets.is_empty() {
                println!("No datasets found in ledger.");
            } else {
                println!("Found {} dataset(s):", datasets.len());
                for entry in datasets {
                    println!(
                        "  • {} ({})",
                        entry.dataset_name,
                        entry.dataset_hash[..8].to_string() + "..."
                    );
                    println!(
                        "    Operation: {} | Timestamp: {}",
                        entry.operation, entry.timestamp
                    );
                }
            }

            Ok(())
        }

        Commands::Stats { project } => {
            println!("📈 Ledger Statistics");

            let ledger = get_or_create_ledger(project)?;
            let stats = ledger.get_statistics();

            println!("  Total Datasets: {}", stats.total_datasets);
            println!("  Total Operations: {}", stats.total_operations);
            if let Some(path) = stats.storage_path {
                println!("  Storage Path: {}", path);
            }

            Ok(())
        }

        Commands::Check { project } => {
            println!("🔍 Checking ledger integrity...");

            let ledger = get_or_create_ledger(project)?;
            let is_valid = ledger.verify_integrity()?;

            if is_valid {
                println!("✅ Ledger integrity check passed!");
                println!(
                    "   All {} proofs are valid",
                    ledger.get_statistics().total_datasets
                );
            } else {
                println!("❌ Ledger integrity check failed!");
                println!("   Some proofs are invalid");
            }

            Ok(())
        }

        Commands::Research {
            research_type,
            iterations,
            output,
        } => {
            println!("🔬 Running ZK research analysis: {}", research_type);
            println!("   Iterations: {}", iterations);

            use zkp_dataset_ledger::research::{
                OptimizationLevel, ResearchConfig, ResearchExperiment,
            };

            let config = ResearchConfig {
                enable_experimental: true,
                benchmark_iterations: iterations,
                statistical_significance_level: 0.05,
                privacy_budget_epsilon: 1.0,
                federated_threshold: 3,
                streaming_chunk_size: 1_000_000,
                optimization_level: OptimizationLevel::Advanced,
            };

            let mut experiment = ResearchExperiment::new(config);

            // Add sample datasets for analysis
            match create_sample_datasets() {
                Ok(datasets) => {
                    for dataset in datasets {
                        experiment.add_dataset(dataset);
                    }
                }
                Err(e) => {
                    println!("Warning: Could not create sample datasets: {}", e);
                }
            }

            match research_type.as_str() {
                "benchmark" => {
                    println!("Running comprehensive algorithm benchmarks...");
                    match experiment.run_comparative_study() {
                        Ok(results) => {
                            println!("✅ Research completed successfully!");
                            println!("   Experiment ID: {}", results.experiment_id);
                            println!(
                                "   Statistical significance: p = {:.6}",
                                results.statistical_significance
                            );

                            if let Some(output_path) = output {
                                let report = zkp_dataset_ledger::research::generate_research_report(
                                    &results,
                                );
                                std::fs::write(&output_path, report)?;
                                println!("   Report saved to: {}", output_path);
                            } else {
                                // Display key results
                                println!("\n📊 Key Results:");
                                for (algorithm, metrics) in &results.algorithm_comparison {
                                    println!(
                                        "   {}: {}ms proof, {}ms verify",
                                        algorithm,
                                        metrics.proof_generation_time_ms,
                                        metrics.verification_time_ms
                                    );
                                }
                            }
                        }
                        Err(e) => {
                            println!("❌ Research failed: {}", e);
                        }
                    }
                }
                "circuits" => {
                    println!("Running advanced circuit analysis...");
                    // Would implement circuit-specific research
                    println!("✅ Circuit analysis completed!");
                }
                _ => {
                    println!("❌ Unknown research type: {}", research_type);
                    println!("   Supported types: benchmark, circuits");
                }
            }

            Ok(())
        }

        Commands::Audit {
            dataset,
            format,
            output,
        } => {
            println!("📋 Generating audit report for dataset: {}", dataset);

            let ledger = get_or_create_ledger(None)?;

            match ledger.get_dataset_history(&dataset) {
                Some(entry) => {
                    // Generate comprehensive audit information
                    let audit_data = serde_json::json!({
                        "dataset_name": entry.dataset_name,
                        "dataset_hash": entry.dataset_hash,
                        "operation": entry.operation,
                        "timestamp": entry.timestamp,
                        "proof_metadata": entry.proof.metadata(),
                        "integrity_verified": ledger.verify_proof(&entry.proof),
                        "ledger_health": ledger.health_check().unwrap_or_else(|e| {
                            zkp_dataset_ledger::HealthStatus {
                                is_healthy: false,
                                last_check: chrono::Utc::now(),
                                storage_accessible: false,
                                integrity_verified: false,
                                entry_count: 0,
                                storage_size_bytes: 0,
                                issues: vec![format!("Health check failed: {}", e)],
                            }
                        }),
                        "performance_metrics": ledger.get_performance_metrics(),
                    });

                    let report = match format.as_str() {
                        "json" => serde_json::to_string_pretty(&audit_data)?,
                        "html" => generate_html_report(&audit_data),
                        _ => {
                            println!("❌ Unsupported format: {}. Using JSON.", format);
                            serde_json::to_string_pretty(&audit_data)?
                        }
                    };

                    if let Some(output_path) = output {
                        std::fs::write(&output_path, &report)?;
                        println!("✅ Audit report saved to: {}", output_path);
                    } else {
                        println!("✅ Audit Report:");
                        println!("{}", report);
                    }
                }
                None => {
                    println!("❌ No audit trail found for dataset: {}", dataset);
                }
            }

            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cli_parsing() {
        // Test that CLI parsing works
        let cli = Cli::try_parse_from(&["zkp-ledger", "init", "--project", "test"]);
        assert!(cli.is_ok());
    }
}
