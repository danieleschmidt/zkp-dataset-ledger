use crate::{Dataset, LedgerError, Proof, ProofConfig, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug)]
pub struct Ledger {
    pub name: String,
    entries: HashMap<String, LedgerEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LedgerEntry {
    pub dataset_name: String,
    pub dataset_hash: String,
    pub proof: Proof,
    pub parent_hash: Option<String>,
    pub operation: Operation,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Operation {
    Notarize,
    Transform { from: String, operation: String },
    Split { ratio: f64, seed: Option<u64> },
}

impl Ledger {
    pub fn new(name: &str) -> Result<Self> {
        Ok(Self {
            name: name.to_string(),
            entries: HashMap::new(),
        })
    }

    pub fn notarize_dataset(
        &mut self,
        dataset: Dataset,
        name: &str,
        config: ProofConfig,
    ) -> Result<Proof> {
        let proof = Proof::generate(&dataset, &config)?;

        let entry = LedgerEntry {
            dataset_name: name.to_string(),
            dataset_hash: dataset.compute_hash(),
            proof: proof.clone(),
            parent_hash: None,
            operation: Operation::Notarize,
            timestamp: chrono::Utc::now(),
        };

        self.entries.insert(name.to_string(), entry);
        Ok(proof)
    }

    pub fn verify_proof(&self, proof: &Proof) -> Result<bool> {
        proof.verify()
    }

    pub fn get_dataset_history(&self, name: &str) -> Result<Vec<&LedgerEntry>> {
        // TODO: Implement proper chain traversal
        Ok(self.entries.get(name).map(|e| vec![e]).unwrap_or_default())
    }
}
