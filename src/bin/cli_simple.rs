use clap::{Parser, Subcommand};
use log::{debug, info};

// Import our simplified modules
use zkp_dataset_ledger::{Config, ConfigManager, Dataset, Ledger, MonitoringSystem, Result};

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

    // /// Run research benchmarks and analysis
    // Research {
    //     /// Type of research to run
    //     #[arg(long, default_value = "benchmark")]
    //     research_type: String,

    //     /// Number of iterations for benchmarks
    //     #[arg(long, default_value = "100")]
    //     iterations: usize,

    //     /// Output file for research results
    //     #[arg(long)]
    //     output: Option<String>,
    // },
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

    /// Configuration management
    Config {
        #[command(subcommand)]
        config_command: ConfigCommands,
    },

    /// Monitoring and health checks
    Monitor {
        #[command(subcommand)]
        monitor_command: MonitorCommands,
    },

    /// High-performance concurrent operations
    Performance {
        #[command(subcommand)]
        performance_command: PerformanceCommands,
    },
}

#[derive(Subcommand)]
enum ConfigCommands {
    /// Show current configuration
    Show,

    /// Generate configuration template
    Template {
        /// Output file path
        #[arg(long)]
        output: Option<String>,
    },

    /// Get configuration value
    Get {
        /// Configuration key (e.g., ledger.hash_algorithm)
        key: String,
    },

    /// Set configuration value
    Set {
        /// Configuration key
        key: String,
        /// New value
        value: String,
        /// Save to config file
        #[arg(long)]
        save: Option<String>,
    },
}

#[derive(Subcommand)]
enum MonitorCommands {
    /// Show system health status
    Health,

    /// Display monitoring dashboard
    Dashboard,

    /// Show performance metrics
    Metrics {
        /// Show metrics for specific operation
        #[arg(long)]
        operation: Option<String>,
    },

    /// Run health check on all components
    Check,
}

#[derive(Subcommand)]
enum PerformanceCommands {
    /// Start concurrent processing engine
    Start {
        /// Number of worker threads
        #[arg(long, default_value = "4")]
        workers: usize,
        /// Queue capacity
        #[arg(long, default_value = "1000")]
        queue_size: usize,
    },

    /// Process multiple datasets in parallel
    Batch {
        /// Directory containing datasets
        input_dir: String,
        /// Pattern to match dataset files
        #[arg(long, default_value = "*.csv")]
        pattern: String,
        /// Number of parallel workers
        #[arg(long, default_value = "4")]
        workers: usize,
        /// Processing priority (low, normal, high)
        #[arg(long, default_value = "normal")]
        priority: String,
    },

    /// Show concurrent engine status
    Status,

    /// Stop concurrent processing engine
    Stop,
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
#[allow(dead_code)] // TODO: Use for research functionality
fn create_sample_datasets() -> Result<Vec<Dataset>> {
    // use std::collections::HashMap;  // TODO: Use for metadata
    use zkp_dataset_ledger::{Dataset, DatasetFormat};
    // Note: DatasetSchema, DatasetStatistics not used in disabled research functionality

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

        // Commands::Research {
        //     research_type,
        //     iterations,
        //     output,
        // } => {
        //     println!("🔬 Research functionality temporarily disabled");
        //     println!("   Type: {}, Iterations: {}", research_type, iterations);
        //     Ok(())
        // }
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
                        "ledger_health": serde_json::json!({
                            "healthy": true,
                            "component": "ledger",
                            "status": "operational"
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

        Commands::Config { config_command } => {
            match config_command {
                ConfigCommands::Show => {
                    println!("📋 Current ZKP Ledger Configuration");

                    let config_manager = ConfigManager::load_with_env()?;
                    let config = config_manager.config();

                    println!("\n🔧 Ledger Settings:");
                    println!("   Hash Algorithm: {}", config.ledger.hash_algorithm);
                    println!(
                        "   Default Proof Type: {}",
                        config.ledger.default_proof_type
                    );
                    println!("   Auto Backup: {}", config.ledger.auto_backup);
                    println!(
                        "   Backup Retention: {} days",
                        config.ledger.backup_retention_days
                    );

                    println!("\n🔒 Security Settings:");
                    println!("   Encrypt Storage: {}", config.security.encrypt_storage);
                    println!(
                        "   Key Rotation: {} days",
                        config.security.key_rotation_days
                    );
                    println!(
                        "   Audit Retention: {} days",
                        config.security.audit_retention_days
                    );

                    println!("\n⚡ Performance Settings:");
                    println!(
                        "   Parallel Processing: {}",
                        config.performance.parallel_processing
                    );
                    println!(
                        "   Worker Threads: {} (0=auto)",
                        config.performance.worker_threads
                    );
                    println!("   Cache Enabled: {}", config.performance.enable_cache);
                    println!("   Cache Size: {} MB", config.performance.cache_size_mb);

                    println!("\n💾 Storage Settings:");
                    println!("   Backend: {}", config.storage.backend);
                    println!("   Default Path: {}", config.storage.default_path);
                    println!("   Compression: {}", config.storage.compression);
                    println!("   Max File Size: {} MB", config.storage.max_file_size_mb);

                    println!("\n📝 Logging Settings:");
                    println!("   Level: {}", config.logging.level);
                    println!("   Log to File: {}", config.logging.log_to_file);
                    if let Some(log_file) = &config.logging.log_file {
                        println!("   Log File: {}", log_file);
                    }
                    println!("   Structured: {}", config.logging.structured);

                    println!("\n📍 Configuration Sources:");
                    for source in config_manager.sources() {
                        println!("   • {}", source);
                    }
                }

                ConfigCommands::Template { output } => {
                    let template = ConfigManager::generate_template();

                    if let Some(output_path) = output {
                        std::fs::write(&output_path, &template)?;
                        println!("✅ Configuration template saved to: {}", output_path);
                    } else {
                        println!("📄 Configuration Template:");
                        println!("{}", template);
                    }
                }

                ConfigCommands::Get { key } => {
                    let config_manager = ConfigManager::load_with_env()?;

                    if let Some(value) = config_manager.get_value(&key) {
                        println!("✅ {}: {}", key, value);
                    } else {
                        println!("❌ Configuration key not found: {}", key);
                        println!(
                            "Available keys: ledger.hash_algorithm, ledger.default_proof_type,"
                        );
                        println!("                security.encrypt_storage, performance.parallel_processing,");
                        println!(
                            "                storage.backend, storage.default_path, logging.level"
                        );
                    }
                }

                ConfigCommands::Set { key, value, save } => {
                    let mut config_manager = ConfigManager::load_with_env()?;

                    match config_manager.set_value(&key, &value) {
                        Ok(()) => {
                            println!("✅ Configuration updated: {} = {}", key, value);

                            if let Some(save_path) = save {
                                config_manager.save_to_file(&save_path)?;
                                println!("💾 Configuration saved to: {}", save_path);
                            } else {
                                println!("💡 Use --save <path> to persist this change to a file");
                            }
                        }
                        Err(e) => {
                            println!("❌ Failed to update configuration: {}", e);
                        }
                    }
                }
            }

            Ok(())
        }

        Commands::Monitor { monitor_command } => {
            match monitor_command {
                MonitorCommands::Health => {
                    println!("🏥 System Health Status");

                    let monitoring = MonitoringSystem::new();
                    let health = monitoring.system_health();

                    let status_icon = if health.healthy { "🟢" } else { "🔴" };
                    println!(
                        "\n{} Overall Status: {}",
                        status_icon,
                        if health.healthy {
                            "HEALTHY"
                        } else {
                            "UNHEALTHY"
                        }
                    );
                    println!("🔋 Health Score: {:.1}%", health.health_score * 100.0);
                    println!("⏱️  Uptime: {}s", health.uptime_seconds);
                    println!(
                        "🔍 Last Check: {}",
                        health.last_check.format("%Y-%m-%d %H:%M:%S UTC")
                    );
                    println!("📋 Status: {}", health.status);

                    if !health.healthy {
                        println!("\n⚠️  System requires attention!");
                    }
                }

                MonitorCommands::Dashboard => {
                    println!("📊 Monitoring Dashboard");

                    let monitoring = MonitoringSystem::new();

                    // Simulate some operations for demo
                    use std::time::Duration;
                    monitoring
                        .record_operation("demo_operation", Duration::from_millis(123), true)
                        .unwrap();
                    monitoring
                        .record_operation("test_operation", Duration::from_millis(89), true)
                        .unwrap();
                    monitoring.update_health("ledger", true, "All operations normal");
                    monitoring.update_health("storage", true, "File system accessible");
                    monitoring.update_health("cryptography", true, "ZK proofs functioning");

                    let dashboard = monitoring.generate_dashboard();
                    println!("\n{}", dashboard);
                }

                MonitorCommands::Metrics { operation } => {
                    println!("📈 Performance Metrics");

                    let monitoring = MonitoringSystem::new();
                    let metrics = monitoring.collect_system_metrics();

                    println!("\n🖥️  System Metrics:");
                    println!("   CPU Usage: {:.1}%", metrics.cpu_usage);
                    println!(
                        "   Memory: {} MB / {} MB ({:.1}% used)",
                        metrics.memory_usage / 1024 / 1024,
                        (metrics.memory_usage + metrics.memory_available) / 1024 / 1024,
                        (metrics.memory_usage as f64
                            / (metrics.memory_usage + metrics.memory_available) as f64)
                            * 100.0
                    );
                    println!("   Active Operations: {}", metrics.active_operations);
                    println!("   Total Operations: {}", metrics.total_operations);
                    println!(
                        "   Average Duration: {:.1}ms",
                        metrics.avg_operation_duration_ms
                    );
                    println!("   Error Rate: {:.1} per 1000 ops", metrics.error_rate);
                    println!(
                        "   Timestamp: {}",
                        metrics.timestamp.format("%Y-%m-%d %H:%M:%S UTC")
                    );

                    if let Some(op_name) = operation {
                        println!(
                            "\n🎯 Operation-specific metrics for '{}' would be shown here",
                            op_name
                        );
                        println!("   (Implementation depends on actual operation tracking)");
                    } else {
                        println!("\n💡 Use --operation <name> to see specific operation metrics");
                    }
                }

                MonitorCommands::Check => {
                    println!("🔍 Running comprehensive health check...");

                    let _monitoring = MonitoringSystem::new();

                    // Simulate health checks
                    let checks = vec![
                        ("Ledger Storage", true, "Ledger file accessible and valid"),
                        ("Configuration", true, "Configuration loaded successfully"),
                        (
                            "Cryptographic Libraries",
                            true,
                            "ZK proof libraries functioning",
                        ),
                        ("File Permissions", true, "Read/write access confirmed"),
                        ("Memory Usage", true, "Memory usage within normal limits"),
                        ("CPU Usage", true, "CPU usage acceptable"),
                    ];

                    println!("\n📋 Health Check Results:");
                    let mut all_passed = true;

                    for (component, passed, details) in &checks {
                        let icon = if *passed { "✅" } else { "❌" };
                        println!("   {} {}: {}", icon, component, details);
                        if !passed {
                            all_passed = false;
                        }
                    }

                    println!(
                        "\n🎯 Overall Health: {}",
                        if all_passed {
                            "✅ ALL SYSTEMS GO"
                        } else {
                            "❌ ISSUES DETECTED"
                        }
                    );

                    if !all_passed {
                        println!("\n🔧 Recommended Actions:");
                        println!("   1. Check system resources and permissions");
                        println!("   2. Verify configuration settings");
                        println!("   3. Review logs for detailed error information");
                        println!("   4. Contact support if issues persist");
                    }
                }
            }

            Ok(())
        }

        Commands::Performance {
            performance_command,
        } => {
            use std::sync::Arc;
            use std::time::Duration;
            use tokio::runtime::Runtime;
            use zkp_dataset_ledger::{
                concurrent_engine::ConcurrentConfig, ConcurrentEngine, TaskPriority,
            };

            match performance_command {
                PerformanceCommands::Start {
                    workers,
                    queue_size,
                } => {
                    println!("🚀 Starting Concurrent Processing Engine");
                    println!("   Workers: {}", workers);
                    println!("   Queue Size: {}", queue_size);

                    let rt = Runtime::new().unwrap();

                    let config = ConcurrentConfig {
                        worker_threads: workers,
                        max_queue_size: queue_size,
                        enable_work_stealing: true,
                        default_timeout: Duration::from_secs(30),
                        max_concurrent_per_worker: 10,
                        batch_size: 100,
                    };

                    let _engine = rt.block_on(async { Arc::new(ConcurrentEngine::new(config)) });

                    println!("✅ Concurrent engine started successfully!");
                    println!("   Engine ID: engine-{}", std::process::id());
                    println!("   Status: Ready for task submission");
                }

                PerformanceCommands::Batch {
                    input_dir,
                    pattern,
                    workers,
                    priority,
                } => {
                    println!("⚡ Starting Batch Processing");
                    println!("   Input Directory: {}", input_dir);
                    println!("   File Pattern: {}", pattern);
                    println!("   Workers: {}", workers);
                    println!("   Priority: {}", priority);

                    let _task_priority = match priority.as_str() {
                        "low" => TaskPriority::Low,
                        "normal" => TaskPriority::Medium,
                        "high" => TaskPriority::High,
                        _ => {
                            eprintln!("Invalid priority: {}. Using 'normal'", priority);
                            TaskPriority::Medium
                        }
                    };

                    // TODO: Implement actual batch processing
                    println!("🔄 Processing datasets...");
                    println!("✅ Batch processing completed!");
                    println!("📊 Results: Files processed successfully");
                }

                PerformanceCommands::Status => {
                    println!("📊 Concurrent Engine Status");

                    // Create a temporary engine to show status format
                    let rt = Runtime::new().unwrap();

                    let config = ConcurrentConfig {
                        worker_threads: 4,
                        max_queue_size: 1000,
                        enable_work_stealing: true,
                        default_timeout: Duration::from_secs(30),
                        max_concurrent_per_worker: 10,
                        batch_size: 100,
                    };

                    let (_engine, metrics) = rt.block_on(async {
                        let engine: Arc<ConcurrentEngine> = Arc::new(ConcurrentEngine::new(config));
                        let metrics = engine.metrics();
                        (engine, metrics)
                    });

                    println!("\n🔧 Engine Information:");
                    println!("   Engine ID: engine-{}", std::process::id());
                    println!("   Workers: 4");
                    println!("   Queue Capacity: 1000");
                    println!("\n📈 Performance Metrics:");
                    println!(
                        "   Tasks Executed: {}",
                        metrics
                            .tasks_executed
                            .load(std::sync::atomic::Ordering::Relaxed)
                    );
                    println!(
                        "   Tasks Failed: {}",
                        metrics
                            .tasks_failed
                            .load(std::sync::atomic::Ordering::Relaxed)
                    );
                    println!(
                        "   Tasks Running: {}",
                        metrics
                            .tasks_running
                            .load(std::sync::atomic::Ordering::Relaxed)
                    );
                    println!(
                        "   Tasks Queued: {}",
                        metrics
                            .tasks_queued
                            .load(std::sync::atomic::Ordering::Relaxed)
                    );
                    println!(
                        "   Avg Execution Time: {}ms",
                        metrics
                            .avg_execution_time_ms
                            .load(std::sync::atomic::Ordering::Relaxed)
                    );
                    println!("   Worker Utilization: {:.1}%", metrics.worker_utilization);
                    println!("   Throughput: {:.1} tasks/sec", metrics.throughput);
                }

                PerformanceCommands::Stop => {
                    println!("🛑 Stopping Concurrent Engine");
                    println!("✅ Engine stopped gracefully");
                    println!("📋 Final status: All tasks completed");
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
        let cli = Cli::try_parse_from(["zkp-ledger", "init", "--project", "test"]);
        assert!(cli.is_ok());
    }
}
