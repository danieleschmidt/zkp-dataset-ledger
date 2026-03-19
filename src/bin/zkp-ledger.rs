//! ZKP Dataset Ledger CLI
//!
//! Usage:
//!   zkp-ledger commit <dataset.csv> [--description "..."]
//!   zkp-ledger prove <commitment.json> <dataset.csv> --property row-count=N
//!   zkp-ledger verify <proof.json>
//!   zkp-ledger ledger init --name "my-ledger"
//!   zkp-ledger ledger add --ledger ledger.json --commitment commitment.json
//!   zkp-ledger ledger verify --ledger ledger.json
//!   zkp-ledger demo

use clap::{Parser, Subcommand};
use std::fs;
use std::path::PathBuf;
use zkp_dataset_ledger::{
    commitment::{CommitmentConfig, DatasetCommitment},
    ledger::DatasetLedger,
    proof::{ProofGenerator, ProofType, ProofVerifier},
};

#[derive(Parser)]
#[command(
    name = "zkp-ledger",
    about = "Zero-knowledge proof ledger for ML dataset provenance",
    version = "0.1.0"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Commit to a dataset (creates a cryptographic commitment)
    Commit {
        /// Path to the dataset file
        dataset: PathBuf,
        /// Human-readable description
        #[arg(short, long)]
        description: Option<String>,
        /// Output file for the commitment JSON
        #[arg(short, long, default_value = "commitment.json")]
        output: PathBuf,
    },
    /// Generate a ZK proof about a dataset property
    Prove {
        /// Path to the commitment JSON (must contain nonce)
        commitment: PathBuf,
        /// Path to the dataset file
        dataset: PathBuf,
        /// Property to prove (e.g., "row-count=3", "min-rows=2", "column-count=5")
        #[arg(short, long)]
        property: String,
        /// Output file for the proof JSON
        #[arg(short, long, default_value = "proof.json")]
        output: PathBuf,
    },
    /// Verify a ZK proof
    Verify {
        /// Path to the proof JSON
        proof: PathBuf,
    },
    /// Ledger management subcommands
    Ledger {
        #[command(subcommand)]
        action: LedgerCommands,
    },
    /// Run an end-to-end demonstration
    Demo,
}

#[derive(Subcommand)]
enum LedgerCommands {
    /// Initialize a new ledger
    Init {
        /// Ledger name
        #[arg(short, long, default_value = "my-ledger")]
        name: String,
        /// Output file
        #[arg(short, long, default_value = "ledger.json")]
        output: PathBuf,
    },
    /// Add a commitment to the ledger
    Add {
        /// Path to the ledger JSON
        #[arg(short, long)]
        ledger: PathBuf,
        /// Path to the commitment JSON
        #[arg(short, long)]
        commitment: PathBuf,
    },
    /// Verify ledger integrity
    Check {
        /// Path to the ledger JSON
        #[arg(short, long)]
        ledger: PathBuf,
    },
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Commit { dataset, description, output } => {
            let data = fs::read(&dataset).unwrap_or_else(|e| {
                eprintln!("Error reading dataset: {e}");
                std::process::exit(1);
            });

            println!("Committing dataset: {}", dataset.display());

            let config = CommitmentConfig {
                include_row_count: true,
                include_column_count: true,
                include_schema_hash: true,
                ..Default::default()
            };

            let commitment = DatasetCommitment::commit(&data, &config, description).unwrap_or_else(|e| {
                eprintln!("Commit failed: {e}");
                std::process::exit(1);
            });

            println!("✓ Dataset hash:   {}", commitment.metadata.dataset_hash);
            println!("✓ Commitment:      {}", commitment.commitment_hash);
            if let Some(rows) = commitment.metadata.row_count {
                println!("✓ Row count:       {rows}");
            }
            if let Some(cols) = commitment.metadata.column_count {
                println!("✓ Column count:    {cols}");
            }
            println!("⚠ Nonce included in output — keep this file private!");

            let json = serde_json::to_string_pretty(&commitment).unwrap();
            fs::write(&output, json).unwrap_or_else(|e| {
                eprintln!("Error writing output: {e}");
                std::process::exit(1);
            });
            println!("→ Commitment saved to {}", output.display());
        }

        Commands::Prove { commitment, dataset, property, output } => {
            let commitment_data = fs::read_to_string(&commitment).unwrap_or_else(|e| {
                eprintln!("Error reading commitment: {e}");
                std::process::exit(1);
            });
            let commitment: DatasetCommitment = serde_json::from_str(&commitment_data).unwrap_or_else(|e| {
                eprintln!("Error parsing commitment: {e}");
                std::process::exit(1);
            });

            let data = fs::read(&dataset).unwrap_or_else(|e| {
                eprintln!("Error reading dataset: {e}");
                std::process::exit(1);
            });

            let proof_type = parse_property(&property).unwrap_or_else(|e| {
                eprintln!("Error parsing property: {e}");
                eprintln!("Valid examples: row-count=3, min-rows=2, max-rows=10, column-count=5, existence");
                std::process::exit(1);
            });

            println!("Generating proof for: {property}");

            let proof = ProofGenerator::generate(&commitment, &data, proof_type).unwrap_or_else(|e| {
                eprintln!("Proof generation failed: {e}");
                std::process::exit(1);
            });

            println!("✓ Challenge:  {}", proof.challenge);
            println!("✓ Response:   {}", proof.response);

            let json = serde_json::to_string_pretty(&proof).unwrap();
            fs::write(&output, json).unwrap_or_else(|e| {
                eprintln!("Error writing output: {e}");
                std::process::exit(1);
            });
            println!("→ Proof saved to {}", output.display());
        }

        Commands::Verify { proof } => {
            let proof_data = fs::read_to_string(&proof).unwrap_or_else(|e| {
                eprintln!("Error reading proof: {e}");
                std::process::exit(1);
            });
            let proof = serde_json::from_str(&proof_data).unwrap_or_else(|e| {
                eprintln!("Error parsing proof: {e}");
                std::process::exit(1);
            });

            match ProofVerifier::verify(&proof) {
                Ok(true) => {
                    println!("✓ Proof VALID");
                    println!("  Commitment: {}", proof.commitment_hash);
                    println!("  Property:   {:?}", proof.proof_type);
                }
                Ok(false) => {
                    println!("✗ Proof INVALID");
                    std::process::exit(1);
                }
                Err(e) => {
                    eprintln!("Verification error: {e}");
                    std::process::exit(1);
                }
            }
        }

        Commands::Ledger { action } => match action {
            LedgerCommands::Init { name, output } => {
                let ledger = DatasetLedger::new(&name);
                ledger.save(&output).unwrap_or_else(|e| {
                    eprintln!("Error saving ledger: {e}");
                    std::process::exit(1);
                });
                println!("✓ Ledger '{name}' initialized → {}", output.display());
            }
            LedgerCommands::Add { ledger, commitment } => {
                let mut l = DatasetLedger::load(&ledger).unwrap_or_else(|e| {
                    eprintln!("Error loading ledger: {e}");
                    std::process::exit(1);
                });
                let c_data = fs::read_to_string(&commitment).unwrap_or_else(|e| {
                    eprintln!("Error reading commitment: {e}");
                    std::process::exit(1);
                });
                let c: DatasetCommitment = serde_json::from_str(&c_data).unwrap_or_else(|e| {
                    eprintln!("Error parsing commitment: {e}");
                    std::process::exit(1);
                });
                let idx = l.append(c).unwrap();
                l.save(&ledger).unwrap();
                println!("✓ Added as entry #{idx}");
            }
            LedgerCommands::Check { ledger } => {
                let l = DatasetLedger::load(&ledger).unwrap_or_else(|e| {
                    eprintln!("Error loading ledger: {e}");
                    std::process::exit(1);
                });
                if l.verify_integrity() {
                    println!("✓ Ledger '{}' integrity OK ({} entries)", l.name, l.len());
                } else {
                    println!("✗ Ledger TAMPERED — integrity check failed");
                    std::process::exit(1);
                }
            }
        },

        Commands::Demo => {
            run_demo();
        }
    }
}

fn parse_property(s: &str) -> Result<ProofType, String> {
    if s == "existence" {
        return Ok(ProofType::Existence);
    }
    if let Some(val) = s.strip_prefix("row-count=") {
        let n = val.parse::<u64>().map_err(|_| format!("invalid number: {val}"))?;
        return Ok(ProofType::ExactRowCount { count: n });
    }
    if let Some(val) = s.strip_prefix("min-rows=") {
        let n = val.parse::<u64>().map_err(|_| format!("invalid number: {val}"))?;
        return Ok(ProofType::MinRowCount { min: n });
    }
    if let Some(val) = s.strip_prefix("max-rows=") {
        let n = val.parse::<u64>().map_err(|_| format!("invalid number: {val}"))?;
        return Ok(ProofType::MaxRowCount { max: n });
    }
    if let Some(val) = s.strip_prefix("column-count=") {
        let n = val.parse::<u64>().map_err(|_| format!("invalid number: {val}"))?;
        return Ok(ProofType::ColumnCount { count: n });
    }
    Err(format!("Unknown property: '{s}'"))
}

fn run_demo() {
    println!("=== ZKP Dataset Ledger Demo ===\n");
    println!("Scenario: A company wants to prove their ML training data has the right");
    println!("properties to an auditor, WITHOUT revealing the actual dataset.\n");

    let dataset = b"employee_id,department,performance_score\n\
                    1001,engineering,87\n\
                    1002,marketing,92\n\
                    1003,engineering,78\n\
                    1004,hr,85\n\
                    1005,marketing,90\n";

    println!("─── Step 1: Data Owner commits to the dataset ───");
    let config = CommitmentConfig {
        include_row_count: true,
        include_column_count: true,
        include_schema_hash: true,
        ..Default::default()
    };
    let commitment = DatasetCommitment::commit(dataset, &config, Some("Employee performance dataset Q4 2025".to_string())).unwrap();
    println!("  Dataset hash:  {}", commitment.metadata.dataset_hash);
    println!("  Commitment:    {}", commitment.commitment_hash);
    println!("  Row count:     {:?}", commitment.metadata.row_count);
    println!("  Column count:  {:?}", commitment.metadata.column_count);
    println!("  [Nonce is kept SECRET by the data owner]");

    println!("\n─── Step 2: Data Owner generates ZK proofs (Auditor receives these) ───");

    let proof_existence = ProofGenerator::generate(&commitment, dataset, ProofType::Existence).unwrap();
    println!("  Proof 1 - Dataset exists:  challenge={}", &proof_existence.challenge[..16]);

    let proof_rows = ProofGenerator::generate(
        &commitment, dataset, ProofType::ExactRowCount { count: 5 }
    ).unwrap();
    println!("  Proof 2 - Exactly 5 rows:  challenge={}", &proof_rows.challenge[..16]);

    let proof_stats = ProofGenerator::generate(
        &commitment,
        dataset,
        ProofType::StatisticalBounds {
            column: "performance_score".to_string(),
            mean_min: 80.0,
            mean_max: 95.0,
        },
    ).unwrap();
    println!("  Proof 3 - Mean score in [80,95]: challenge={}", &proof_stats.challenge[..16]);

    println!("\n─── Step 3: Auditor verifies proofs (WITHOUT seeing the dataset) ───");

    let v1 = ProofVerifier::verify(&proof_existence).unwrap();
    let v2 = ProofVerifier::verify(&proof_rows).unwrap();
    let v3 = ProofVerifier::verify(&proof_stats).unwrap();

    println!("  Proof 1 (existence):     {}", if v1 { "✓ VALID" } else { "✗ INVALID" });
    println!("  Proof 2 (row count = 5): {}", if v2 { "✓ VALID" } else { "✗ INVALID" });
    println!("  Proof 3 (mean ∈ [80,95]):{}", if v3 { "✓ VALID" } else { "✗ INVALID" });

    println!("\n─── Step 4: Tamper detection ───");
    let mut tampered = proof_rows.clone();
    tampered.proof_type = ProofType::ExactRowCount { count: 9999 };
    let vtampered = ProofVerifier::verify(&tampered).unwrap();
    println!("  Tampered proof (claimed 9999 rows): {}", if vtampered { "✓ VALID" } else { "✗ REJECTED (correct!)" });

    println!("\n─── Step 5: Append to ledger ───");
    let mut ledger = DatasetLedger::new("employee-data-audit");
    let idx = ledger.append_with_proofs(
        commitment,
        vec![proof_existence, proof_rows, proof_stats],
    ).unwrap();
    println!("  Entry #{idx} appended");
    println!("  Ledger integrity: {}", if ledger.verify_integrity() { "✓ OK" } else { "✗ FAILED" });

    println!("\n=== Summary ===");
    println!("The auditor can verify all properties of the dataset");
    println!("without ever seeing the actual employee records.");
    println!("This is the core value of ZKP for ML provenance.");
}
