use clap::{Parser, Subcommand};
use std::path::PathBuf;

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
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    if cli.verbose {
        println!("ZKP Dataset Ledger - Simplified Version");
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

        Commands::Notarize { dataset, name, proof_type } => {
            println!("Notarizing dataset: {} ({})", name, dataset);
            
            let mut ledger = Ledger::new("default".to_string());
            
            // Check if file exists
            if !std::path::Path::new(&dataset).exists() {
                return Err(zkp_dataset_ledger::LedgerError::not_found(
                    format!("Dataset file not found: {}", dataset)
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
            
            let ledger = Ledger::new("default".to_string());
            
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
            
            let ledger = Ledger::new("default".to_string());
            
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