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
