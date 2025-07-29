use clap::{Parser, Subcommand};
use zkp_dataset_ledger::{Ledger, Dataset, ProofConfig, Result};

#[derive(Parser)]
#[command(name = "zkp-ledger")]
#[command(about = "A CLI for ZKP Dataset Ledger")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Init {
        #[arg(long)]
        project: String,
    },
    Notarize {
        dataset: String,
        #[arg(long)]
        name: String,
    },
    Verify {
        proof_file: String,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::init();
    
    let cli = Cli::parse();
    
    match cli.command {
        Commands::Init { project } => {
            let _ledger = Ledger::new(&project)?;
            println!("Initialized ledger for project: {}", project);
        }
        Commands::Notarize { dataset, name } => {
            let mut ledger = Ledger::new("default")?;
            let dataset = Dataset::from_path(&dataset)?;
            let proof = ledger.notarize_dataset(dataset, &name, ProofConfig::default())?;
            println!("Notarized dataset '{}' with proof size: {} bytes", name, proof.size_bytes());
        }
        Commands::Verify { proof_file: _ } => {
            println!("Proof verification not yet implemented");
        }
    }
    
    Ok(())
}