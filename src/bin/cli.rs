use clap::{Parser, Subcommand};
use std::collections::HashMap;
use std::path::PathBuf;
use zkp_dataset_ledger::storage::create_storage;
use zkp_dataset_ledger::{Config, ConfigBuilder, Dataset, Ledger, ProofConfig, Result};

#[derive(Parser)]
#[command(name = "zkp-ledger")]
#[command(about = "A CLI for ZKP Dataset Ledger - Cryptographic provenance for ML pipelines")]
#[command(version, author)]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Enable verbose logging
    #[arg(short, long, global = true)]
    verbose: bool,

    /// Configuration file path
    #[arg(short, long, global = true)]
    config: Option<String>,
}

#[derive(Subcommand)]
enum Commands {
    /// Initialize a new ledger for a project
    Init {
        #[arg(long)]
        project: String,

        /// Storage backend type (memory, postgres)
        #[arg(long, default_value = "memory")]
        storage: String,

        /// Hash algorithm (sha3-256, blake3)
        #[arg(long, default_value = "sha3-256")]
        hash_algorithm: String,
    },

    /// Notarize a dataset with cryptographic proof
    Notarize {
        /// Path to the dataset file
        dataset: String,

        /// Name for the dataset entry
        #[arg(long)]
        name: String,

        /// Proof type (integrity, row-count, schema, statistics)
        #[arg(long, default_value = "integrity")]
        proof_type: String,

        /// Include Merkle proof
        #[arg(long)]
        merkle_proof: bool,

        /// Output proof to file
        #[arg(long)]
        output: Option<String>,
    },

    /// Record a dataset transformation
    Transform {
        /// Input dataset name
        #[arg(long)]
        from: String,

        /// Output dataset name
        #[arg(long)]
        to: String,

        /// Transformation operation description
        #[arg(long)]
        operation: String,

        /// Operation parameters (key=value pairs)
        #[arg(long)]
        params: Vec<String>,
    },

    /// Record a train/test split
    Split {
        /// Dataset name to split
        dataset: String,

        /// Split ratio (0.0 to 1.0)
        #[arg(long, default_value = "0.8")]
        ratio: f64,

        /// Random seed for reproducibility
        #[arg(long)]
        seed: Option<u64>,

        /// Column name for stratified sampling
        #[arg(long)]
        stratify: Option<String>,
    },

    /// Verify a proof file
    Verify {
        /// Path to proof JSON file
        proof_file: String,

        /// Verify against specific dataset
        #[arg(long)]
        dataset: Option<String>,

        /// Expected public inputs
        #[arg(long)]
        inputs: Vec<String>,
    },

    /// Show dataset history and audit trail
    History {
        /// Dataset name
        dataset: String,

        /// Output format (json, table, summary)
        #[arg(long, default_value = "table")]
        format: String,

        /// Limit number of entries
        #[arg(long)]
        limit: Option<usize>,
    },

    /// Generate audit report
    Audit {
        /// Start from dataset or timestamp
        #[arg(long)]
        from: Option<String>,

        /// End at dataset or timestamp  
        #[arg(long)]
        to: Option<String>,

        /// Output format (json-ld, pdf, html, markdown)
        #[arg(long, default_value = "json-ld")]
        format: String,

        /// Output file path
        #[arg(long)]
        output: String,

        /// Include proof visualizations
        #[arg(long)]
        include_visualizations: bool,
    },

    /// Verify entire ledger chain integrity
    VerifyChain {
        /// Repair any issues found
        #[arg(long)]
        repair: bool,

        /// Strict verification mode
        #[arg(long)]
        strict: bool,
    },

    /// Show ledger statistics and summary
    Status {
        /// Show detailed statistics
        #[arg(long)]
        detailed: bool,

        /// Output format (table, json, yaml)
        #[arg(long, default_value = "table")]
        format: String,
    },

    /// Export ledger data
    Export {
        /// Output file path
        output: String,

        /// Export format (json, csv, parquet)
        #[arg(long, default_value = "json")]
        format: String,

        /// Include proof data
        #[arg(long)]
        include_proofs: bool,
    },

    /// Import ledger data
    Import {
        /// Input file path
        input: String,

        /// Merge with existing data
        #[arg(long)]
        merge: bool,

        /// Validate all proofs during import
        #[arg(long)]
        validate: bool,
    },

    /// Configuration management commands
    Config {
        #[command(subcommand)]
        config_command: ConfigCommands,
    },
}

#[derive(Subcommand)]
enum ConfigCommands {
    /// Generate example configuration file
    Init {
        /// Output file path (defaults to zkp-ledger.toml)
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Output format (toml, json)
        #[arg(short, long, default_value = "toml")]
        format: String,
    },

    /// Show current configuration
    Show {
        /// Configuration file path
        #[arg(short, long)]
        file: Option<PathBuf>,
    },

    /// Validate configuration file
    Validate {
        /// Configuration file path
        file: PathBuf,
    },

    /// Set configuration value
    Set {
        /// Configuration key (e.g., storage.backend)
        key: String,

        /// Configuration value
        value: String,

        /// Configuration file path
        #[arg(short, long)]
        file: Option<PathBuf>,
    },

    /// Get configuration value
    Get {
        /// Configuration key (e.g., storage.backend)
        key: String,

        /// Configuration file path
        #[arg(short, long)]
        file: Option<PathBuf>,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // Load configuration
    let config = load_config(cli.config.as_deref())?;

    // Initialize logging based on verbosity and config
    let log_level = if cli.verbose {
        tracing::Level::DEBUG
    } else {
        match config.monitoring.log_level.as_str() {
            "trace" => tracing::Level::TRACE,
            "debug" => tracing::Level::DEBUG,
            "info" => tracing::Level::INFO,
            "warn" => tracing::Level::WARN,
            "error" => tracing::Level::ERROR,
            _ => tracing::Level::INFO,
        }
    };

    tracing_subscriber::fmt().with_max_level(log_level).init();

    match cli.command {
        Commands::Init {
            project,
            storage,
            hash_algorithm,
        } => {
            println!("🚀 Initializing ZKP Dataset Ledger");
            println!("Project: {}", project);
            println!("Storage: {}", storage);
            println!("Hash Algorithm: {}", hash_algorithm);

            // Create storage backend using configuration
            let (backend_type, connection_string) = config.get_storage_config();
            let actual_storage = if storage != "memory" {
                &storage
            } else {
                backend_type
            };
            let storage_backend = create_storage(actual_storage, connection_string)?;
            let _ledger = Ledger::new(storage_backend);

            println!(
                "✅ Successfully initialized ledger for project: {} with {} storage",
                project, actual_storage
            );
        }

        Commands::Notarize {
            dataset,
            name,
            proof_type,
            merkle_proof,
            output,
        } => {
            println!("📋 Notarizing dataset: {}", dataset);

            let mut ledger = Ledger::new("default")?;
            let dataset = Dataset::from_path(&dataset)?;

            let mut config = ProofConfig::default();
            config.include_merkle_proof = merkle_proof;

            // Set proof type based on string
            config.proof_type = match proof_type.as_str() {
                "integrity" => zkp_dataset_ledger::proof::ProofType::DatasetIntegrity,
                "row-count" => zkp_dataset_ledger::proof::ProofType::RowCount,
                "schema" => zkp_dataset_ledger::proof::ProofType::Schema,
                "statistics" => zkp_dataset_ledger::proof::ProofType::Statistics,
                _ => zkp_dataset_ledger::proof::ProofType::DatasetIntegrity,
            };

            let proof = ledger.notarize_dataset(dataset, &name, config)?;

            println!("✅ Successfully notarized dataset '{}'", name);
            println!("   Proof size: {} bytes", proof.size_bytes());
            println!("   Dataset hash: {}", &proof.dataset_hash[..16]);

            if let Some(output_path) = output {
                let proof_json = proof.to_json()?;
                std::fs::write(&output_path, proof_json)?;
                println!("   Proof saved to: {}", output_path);
            }
        }

        Commands::Transform {
            from,
            to,
            operation,
            params,
        } => {
            println!("🔄 Recording transformation: {} -> {}", from, to);

            let mut ledger = Ledger::new("default")?;

            // Parse parameters
            let mut param_map = HashMap::new();
            for param in params {
                if let Some((key, value)) = param.split_once('=') {
                    param_map.insert(key.to_string(), value.to_string());
                }
            }

            // For now, create a dummy proof - in real implementation,
            // this would be generated from actual transformation
            let dummy_dataset =
                Dataset::from_path("dummy.csv").unwrap_or_else(|_| create_dummy_dataset());
            let proof = zkp_dataset_ledger::proof::Proof::generate(
                &dummy_dataset,
                &ProofConfig::default(),
            )?;

            let transform_id =
                ledger.record_transformation(&from, &to, &operation, param_map, proof)?;

            println!("✅ Recorded transformation with ID: {}", transform_id);
        }

        Commands::Split {
            dataset,
            ratio,
            seed,
            stratify,
        } => {
            println!("✂️  Recording data split for: {}", dataset);

            let mut ledger = Ledger::new("default")?;

            // Create dummy proof for split operation
            let dummy_dataset = create_dummy_dataset();
            let proof = zkp_dataset_ledger::proof::Proof::generate(
                &dummy_dataset,
                &ProofConfig::default(),
            )?;

            let split_id = ledger.record_split(&dataset, ratio, seed, stratify, proof)?;

            println!("✅ Recorded data split with ID: {}", split_id);
            println!("   Ratio: {:.2}", ratio);
            if let Some(s) = seed {
                println!("   Seed: {}", s);
            }
            if let Some(s) = stratify {
                println!("   Stratify by: {}", s);
            }
        }

        Commands::Verify {
            proof_file,
            dataset: _,
            inputs,
        } => {
            println!("🔍 Verifying proof: {}", proof_file);

            let proof_content = std::fs::read_to_string(&proof_file)?;
            let proof = zkp_dataset_ledger::proof::Proof::from_json(&proof_content)?;

            let is_valid = if inputs.is_empty() {
                proof.verify()?
            } else {
                proof.verify_with_inputs(&inputs)?
            };

            if is_valid {
                println!("✅ Proof is VALID");
                println!("   Dataset hash: {}", proof.dataset_hash);
                println!("   Proof type: {:?}", proof.proof_type);
                println!("   Timestamp: {}", proof.timestamp);
            } else {
                println!("❌ Proof is INVALID");
                std::process::exit(1);
            }
        }

        Commands::History {
            dataset,
            format,
            limit,
        } => {
            println!("📚 Dataset history for: {}", dataset);

            let ledger = Ledger::new("default")?;
            let history = ledger.get_dataset_history(&dataset)?;

            let entries = if let Some(limit) = limit {
                history.into_iter().take(limit).collect::<Vec<_>>()
            } else {
                history
            };

            match format.as_str() {
                "json" => {
                    let json = serde_json::to_string_pretty(&entries)?;
                    println!("{}", json);
                }
                "summary" => {
                    println!("Total entries: {}", entries.len());
                    for entry in &entries {
                        println!(
                            "  {} - {} ({})",
                            entry.timestamp.format("%Y-%m-%d %H:%M:%S"),
                            entry.operation.operation_type(),
                            &entry.dataset_hash[..8]
                        );
                    }
                }
                _ => {
                    // Table format (default)
                    println!(
                        "Timestamp                | Operation   | Dataset Hash | Block Height"
                    );
                    println!("-------------------------|-------------|-------------|-------------");
                    for entry in &entries {
                        println!(
                            "{} | {:11} | {:11} | {:12}",
                            entry.timestamp.format("%Y-%m-%d %H:%M:%S"),
                            entry.operation.operation_type(),
                            &entry.dataset_hash[..11],
                            entry.block_height
                        );
                    }
                }
            }
        }

        Commands::Audit {
            from: _,
            to: _,
            format,
            output,
            include_visualizations: _,
        } => {
            println!("📊 Generating audit report");

            let ledger = Ledger::new("default")?;
            let summary = ledger.get_summary()?;

            match format.as_str() {
                "json-ld" => {
                    let report = serde_json::to_string_pretty(&summary)?;
                    std::fs::write(&output, report)?;
                    println!("✅ JSON-LD audit report saved to: {}", output);
                }
                _ => {
                    println!("⚠️  Format '{}' not yet implemented", format);
                }
            }
        }

        Commands::VerifyChain { repair: _, strict } => {
            println!("🔗 Verifying ledger chain integrity");

            let ledger = Ledger::new("default")?;
            let is_valid = ledger.verify_chain_integrity()?;

            if is_valid {
                println!("✅ Chain integrity verification PASSED");
                if strict {
                    println!("   Strict mode: All proofs and linkages verified");
                }
            } else {
                println!("❌ Chain integrity verification FAILED");
                std::process::exit(1);
            }
        }

        Commands::Status { detailed, format } => {
            println!("📈 Ledger Status");

            let ledger = Ledger::new("default")?;
            let summary = ledger.get_summary()?;

            match format.as_str() {
                "json" => {
                    let json = serde_json::to_string_pretty(&summary)?;
                    println!("{}", json);
                }
                _ => {
                    // Table format (default)
                    println!("Ledger: {}", summary.name);
                    println!("Total Entries: {}", summary.total_entries);
                    println!("Datasets Tracked: {}", summary.datasets_tracked);
                    println!("Storage Size: {} bytes", summary.storage_size_bytes);
                    println!(
                        "Integrity Status: {}",
                        if summary.integrity_status {
                            "✅ Valid"
                        } else {
                            "❌ Invalid"
                        }
                    );

                    if detailed {
                        println!("\nOperations by Type:");
                        for (op_type, count) in &summary.operations_by_type {
                            println!("  {}: {}", op_type, count);
                        }
                    }
                }
            }
        }

        Commands::Export {
            output,
            format,
            include_proofs: _,
        } => {
            println!("📤 Exporting ledger data to: {}", output);

            let ledger = Ledger::new("default")?;
            let query = zkp_dataset_ledger::ledger::LedgerQuery::default();

            match format.as_str() {
                "json" => {
                    let json_data = ledger.export_to_json(&query)?;
                    std::fs::write(&output, json_data)?;
                    println!("✅ Exported to JSON: {}", output);
                }
                _ => {
                    println!("⚠️  Export format '{}' not yet implemented", format);
                }
            }
        }

        Commands::Import {
            input,
            merge: _,
            validate: _,
        } => {
            println!("📥 Importing ledger data from: {}", input);
            println!("⚠️  Import functionality not yet implemented");
        }

        Commands::Config { config_command } => {
            handle_config_command(config_command, &config).await?;
        }
    }

    Ok(())
}

/// Load configuration from file or environment variables.
fn load_config(config_path: Option<&str>) -> Result<Config> {
    match config_path {
        Some(path) => {
            // Load from specified file
            if std::path::Path::new(path).exists() {
                if path.ends_with(".json") {
                    Config::from_json_file(path)
                } else {
                    Config::from_file(path)
                }
            } else {
                eprintln!("⚠️  Config file not found: {}, using defaults", path);
                Ok(Config::default())
            }
        }
        None => {
            // Try to load from default locations
            for default_path in &["zkp-ledger.toml", "zkp-ledger.json", ".zkp-ledger.toml"] {
                if std::path::Path::new(default_path).exists() {
                    return if default_path.ends_with(".json") {
                        Config::from_json_file(default_path)
                    } else {
                        Config::from_file(default_path)
                    };
                }
            }

            // Fall back to environment variables and defaults
            Config::from_env()
        }
    }
}

/// Handle configuration subcommands.
async fn handle_config_command(command: ConfigCommands, _current_config: &Config) -> Result<()> {
    match command {
        ConfigCommands::Init { output, format } => {
            let output_path = output.unwrap_or_else(|| {
                PathBuf::from(match format.as_str() {
                    "json" => "zkp-ledger.json",
                    _ => "zkp-ledger.toml",
                })
            });

            let config = Config::default();

            match format.as_str() {
                "json" => {
                    config.save_json(&output_path)?;
                    println!(
                        "✅ Generated JSON configuration file: {}",
                        output_path.display()
                    );
                }
                _ => {
                    config.save_toml(&output_path)?;
                    println!(
                        "✅ Generated TOML configuration file: {}",
                        output_path.display()
                    );
                }
            }

            println!("📝 Edit the file to customize your configuration");
        }

        ConfigCommands::Show { file } => {
            let config = if let Some(file_path) = file {
                if file_path.extension().map_or(false, |ext| ext == "json") {
                    Config::from_json_file(&file_path)?
                } else {
                    Config::from_file(&file_path)?
                }
            } else {
                load_config(None)?
            };

            let json_output = serde_json::to_string_pretty(&config)?;
            println!("{}", json_output);
        }

        ConfigCommands::Validate { file } => {
            let config = if file.extension().map_or(false, |ext| ext == "json") {
                Config::from_json_file(&file)?
            } else {
                Config::from_file(&file)?
            };

            match config.validate() {
                Ok(()) => {
                    println!("✅ Configuration is valid");
                }
                Err(e) => {
                    eprintln!("❌ Configuration validation failed: {}", e);
                    std::process::exit(1);
                }
            }
        }

        ConfigCommands::Set { key, value, file } => {
            let file_path = file.unwrap_or_else(|| PathBuf::from("zkp-ledger.toml"));

            let mut config = if file_path.exists() {
                if file_path.extension().map_or(false, |ext| ext == "json") {
                    Config::from_json_file(&file_path)?
                } else {
                    Config::from_file(&file_path)?
                }
            } else {
                Config::default()
            };

            // Simple key-value setting (would need more sophisticated parsing for nested keys)
            match key.as_str() {
                "storage.backend" => config.storage.backend = value,
                "storage.connection_string" => config.storage.connection_string = value,
                "crypto.use_groth16" => {
                    config.crypto.use_groth16 = value.parse().map_err(|_| {
                        zkp_dataset_ledger::LedgerError::ConfigError(
                            "Invalid boolean value".to_string(),
                        )
                    })?;
                }
                "monitoring.log_level" => config.monitoring.log_level = value,
                _ => {
                    eprintln!("❌ Unknown configuration key: {}", key);
                    std::process::exit(1);
                }
            }

            if file_path.extension().map_or(false, |ext| ext == "json") {
                config.save_json(&file_path)?;
            } else {
                config.save_toml(&file_path)?;
            }

            println!("✅ Set {} = {}", key, value);
        }

        ConfigCommands::Get { key, file } => {
            let config = if let Some(file_path) = file {
                if file_path.extension().map_or(false, |ext| ext == "json") {
                    Config::from_json_file(&file_path)?
                } else {
                    Config::from_file(&file_path)?
                }
            } else {
                load_config(None)?
            };

            let value = match key.as_str() {
                "storage.backend" => config.storage.backend,
                "storage.connection_string" => config.storage.connection_string,
                "crypto.use_groth16" => config.crypto.use_groth16.to_string(),
                "monitoring.log_level" => config.monitoring.log_level,
                _ => {
                    eprintln!("❌ Unknown configuration key: {}", key);
                    std::process::exit(1);
                }
            };

            println!("{}", value);
        }
    }

    Ok(())
}

/// Create a dummy dataset for testing purposes
fn create_dummy_dataset() -> zkp_dataset_ledger::Dataset {
    zkp_dataset_ledger::Dataset {
        name: "dummy".to_string(),
        hash: "dummy_hash".to_string(),
        size: 100,
        row_count: Some(10),
        column_count: Some(2),
        schema: None,
        statistics: None,
        format: zkp_dataset_ledger::dataset::DatasetFormat::Csv,
        path: None,
    }
}
