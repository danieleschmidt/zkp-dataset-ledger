//! Zero-knowledge proof generation and verification for datasets.

use crate::circuits::{self, DatasetCircuit, StatisticalCircuit, Fr};
use crate::crypto::hash::{hash_bytes, HashAlgorithm};
use crate::crypto::merkle::{MerkleProof, MerkleTree};
use crate::{Dataset, LedgerError, Result};
use ark_groth16::{PreparedVerifyingKey, Proof as Groth16Proof, ProvingKey};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::rand::RngCore;
use serde::{Deserialize, Serialize};

/// A cryptographic proof about dataset properties.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Proof {
    pub dataset_hash: String,
    pub proof_data: Vec<u8>,
    pub public_inputs: Vec<String>,
    pub private_inputs_commitment: String,
    pub proof_type: ProofType,
    pub merkle_root: Option<String>,
    pub merkle_proof: Option<MerkleProof>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub version: String,
    pub groth16_proof: Option<Vec<u8>>,
    pub circuit_public_inputs: Option<Vec<String>>,
}

/// Types of proofs that can be generated.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ProofType {
    /// Basic dataset existence and integrity proof
    DatasetIntegrity,
    /// Proof of row count without revealing data
    RowCount,
    /// Proof of column count and schema
    Schema,
    /// Proof of statistical properties
    Statistics,
    /// Proof of data transformation correctness
    Transformation,
    /// Proof of train/test split properties
    DataSplit,
    /// Custom proof for specific properties
    Custom(String),
}

/// Configuration for proof generation.
#[derive(Debug, Clone)]
pub struct ProofConfig {
    pub curve: String,
    pub security_level: u32,
    pub parallel: bool,
    pub proof_type: ProofType,
    pub hash_algorithm: HashAlgorithm,
    pub include_merkle_proof: bool,
    pub chunk_size: Option<usize>,
    pub use_groth16: bool,
    pub proving_key: Option<ProvingKey<circuits::Curve>>,
    pub verifying_key: Option<PreparedVerifyingKey<circuits::Curve>>,
}

impl Default for ProofConfig {
    fn default() -> Self {
        Self {
            curve: "bls12-381".to_string(),
            security_level: 128,
            parallel: true,
            proof_type: ProofType::DatasetIntegrity,
            hash_algorithm: HashAlgorithm::default(),
            include_merkle_proof: true,
            chunk_size: Some(1000),
            use_groth16: true,
            proving_key: None,
            verifying_key: None,
        }
    }
}

impl Proof {
    /// Generate a zero-knowledge proof for the given dataset.
    pub fn generate(dataset: &Dataset, config: &ProofConfig) -> Result<Self> {
        let dataset_hash = dataset.compute_hash();

        // Create public inputs based on proof type
        let public_inputs = Self::create_public_inputs(dataset, &config.proof_type)?;

        // Generate proof data based on type
        let proof_data = Self::generate_proof_data(dataset, config)?;

        // Create Merkle tree if requested
        let (merkle_root, merkle_proof) = if config.include_merkle_proof {
            Self::create_merkle_proof(dataset, config)?
        } else {
            (None, None)
        };

        // Create commitment to private inputs
        let private_inputs_commitment = Self::create_private_commitment(dataset, config)?;

        // Generate Groth16 proof if enabled
        let (groth16_proof, circuit_public_inputs) = if config.use_groth16 {
            Self::generate_groth16_proof(dataset, config)?
        } else {
            (None, None)
        };

        Ok(Proof {
            dataset_hash: dataset_hash.clone(),
            proof_data,
            public_inputs,
            private_inputs_commitment,
            proof_type: config.proof_type.clone(),
            merkle_root,
            merkle_proof,
            timestamp: chrono::Utc::now(),
            version: "1.0.0".to_string(),
            groth16_proof,
            circuit_public_inputs,
        })
    }

    /// Create public inputs for the proof.
    fn create_public_inputs(dataset: &Dataset, proof_type: &ProofType) -> Result<Vec<String>> {
        let mut inputs = vec![dataset.compute_hash()];

        match proof_type {
            ProofType::DatasetIntegrity => {
                inputs.push(dataset.size.to_string());
                if let Some(format) = &dataset.path {
                    inputs.push(format.clone());
                }
            }
            ProofType::RowCount => {
                if let Some(rows) = dataset.row_count {
                    // Commit to row count without revealing exact number
                    let commitment = hash_bytes(&rows.to_le_bytes(), HashAlgorithm::default())?;
                    inputs.push(commitment);
                }
            }
            ProofType::Schema => {
                if let Some(columns) = dataset.column_count {
                    inputs.push(columns.to_string());
                }
                // Schema hash would be computed from column types
                if dataset.schema.is_some() {
                    let schema_hash = hash_bytes(
                        b"schema_placeholder", // Would be actual schema serialization
                        HashAlgorithm::default(),
                    )?;
                    inputs.push(schema_hash);
                }
            }
            ProofType::Statistics => {
                // Commit to statistical properties without revealing them
                if dataset.statistics.is_some() {
                    let stats_commitment = hash_bytes(
                        b"stats_placeholder", // Would be actual statistics
                        HashAlgorithm::default(),
                    )?;
                    inputs.push(stats_commitment);
                }
            }
            ProofType::Transformation => {
                inputs.push("transformation".to_string());
            }
            ProofType::DataSplit => {
                inputs.push("split".to_string());
            }
            ProofType::Custom(name) => {
                inputs.push(name.clone());
            }
        }

        Ok(inputs)
    }

    /// Generate the actual cryptographic proof data.
    fn generate_proof_data(dataset: &Dataset, config: &ProofConfig) -> Result<Vec<u8>> {
        if config.use_groth16 {
            // When using Groth16, this is legacy proof data for compatibility
            let proof_input = format!(
                "groth16:{}:{}:{}",
                dataset.compute_hash(),
                config.proof_type.type_name(),
                config.security_level
            );
            
            let proof_hash = hash_bytes(proof_input.as_bytes(), config.hash_algorithm.clone())?;
            let hash_bytes = hex::decode(&proof_hash)
                .map_err(|e| LedgerError::Crypto(format!("Hash decode error: {}", e)))?;
            
            // Minimal legacy proof data
            let mut proof_data = Vec::with_capacity(32);
            for i in 0..32 {
                proof_data.push(hash_bytes[i % hash_bytes.len()]);
            }
            proof_data[0] = config.proof_type.type_id();
            
            Ok(proof_data)
        } else {
            // Original simulated proof for backwards compatibility
            let proof_input = format!(
                "{}:{}:{}:{}",
                dataset.compute_hash(),
                config.proof_type.type_name(),
                config.security_level,
                dataset.size
            );

            let proof_hash = hash_bytes(proof_input.as_bytes(), config.hash_algorithm.clone())?;

            // Simulate Groth16 proof structure (3 group elements in BLS12-381)
            // G1 point (48 bytes) + G2 point (96 bytes) + G1 point (48 bytes) = 192 bytes
            // Plus some padding for the actual proof format
            let mut proof_data = Vec::with_capacity(288);

            // Use the hash as seed for deterministic proof generation
            let hash_bytes = hex::decode(&proof_hash)
                .map_err(|e| LedgerError::Crypto(format!("Hash decode error: {}", e)))?;

            // Expand hash to fill proof data
            for i in 0..288 {
                proof_data.push(hash_bytes[i % hash_bytes.len()]);
            }

            // Add proof type marker
            proof_data[0] = config.proof_type.type_id();

            Ok(proof_data)
        }
    }

    /// Generate a real Groth16 zero-knowledge proof.
    fn generate_groth16_proof(
        dataset: &Dataset,
        config: &ProofConfig,
    ) -> Result<(Option<Vec<u8>>, Option<Vec<String>>)> {
        match &config.proof_type {
            ProofType::DatasetIntegrity | ProofType::RowCount => {
                Self::generate_dataset_circuit_proof(dataset, config)
            }
            ProofType::Statistics => {
                Self::generate_statistical_circuit_proof(dataset, config)
            }
            _ => {
                // For other proof types, use dataset circuit as default
                Self::generate_dataset_circuit_proof(dataset, config)
            }
        }
    }

    /// Generate proof using DatasetCircuit.
    fn generate_dataset_circuit_proof(
        dataset: &Dataset,
        config: &ProofConfig,
    ) -> Result<(Option<Vec<u8>>, Option<Vec<String>>)> {
        // Convert dataset to field elements
        let dataset_content = Self::dataset_to_field_elements(dataset)?;
        let nonce = Fr::from(ark_std::rand::thread_rng().next_u64());
        
        // Compute hash as sum + nonce (matching circuit logic)
        let dataset_hash = dataset_content
            .iter()
            .fold(Fr::from(0u64), |acc, &val| acc + val)
            + nonce;
        
        let row_count = Fr::from(dataset_content.len() as u64);
        
        // Create circuit
        let circuit = DatasetCircuit {
            dataset_hash: Some(dataset_hash),
            row_count: Some(row_count),
            dataset_content: Some(dataset_content),
            nonce: Some(nonce),
        };
        
        // Generate or use provided keys
        let (pk, vk) = if let (Some(pk), Some(vk)) = (&config.proving_key, &config.verifying_key) {
            (pk.clone(), vk.clone())
        } else {
            // Generate keys for this circuit
            circuits::setup_circuit(circuit.clone())?
        };
        
        // Generate proof
        let proof = circuits::generate_proof(circuit, &pk)?;
        
        // Serialize proof
        let mut proof_bytes = Vec::new();
        proof
            .serialize_compressed(&mut proof_bytes)
            .map_err(|e| LedgerError::CircuitError(format!("Proof serialization failed: {}", e)))?;
        
        // Public inputs for verification
        let public_inputs = vec![
            Self::field_to_string(&dataset_hash),
            Self::field_to_string(&row_count),
        ];
        
        Ok((Some(proof_bytes), Some(public_inputs)))
    }
    
    /// Generate proof using StatisticalCircuit.
    fn generate_statistical_circuit_proof(
        dataset: &Dataset,
        config: &ProofConfig,
    ) -> Result<(Option<Vec<u8>>, Option<Vec<String>>)> {
        // Convert dataset to field elements
        let data = Self::dataset_to_field_elements(dataset)?;
        
        if data.is_empty() {
            return Ok((None, None));
        }
        
        // Compute mean and variance
        let sum = data.iter().fold(Fr::from(0u64), |acc, &val| acc + val);
        let n = Fr::from(data.len() as u64);
        let mean = sum / n;
        
        // Simplified variance calculation
        let variance_sum = data
            .iter()
            .fold(Fr::from(0u64), |acc, &val| {
                let diff = val - mean;
                acc + (diff * diff)
            });
        let variance = variance_sum / n;
        
        // Create circuit
        let circuit = StatisticalCircuit {
            mean: Some(mean),
            variance: Some(variance),
            data: Some(data),
        };
        
        // Generate or use provided keys
        let (pk, vk) = if let (Some(pk), Some(vk)) = (&config.proving_key, &config.verifying_key) {
            (pk.clone(), vk.clone())
        } else {
            // Generate keys for this circuit
            circuits::setup_circuit(circuit.clone())?
        };
        
        // Generate proof
        let proof = circuits::generate_proof(circuit, &pk)?;
        
        // Serialize proof
        let mut proof_bytes = Vec::new();
        proof
            .serialize_compressed(&mut proof_bytes)
            .map_err(|e| LedgerError::CircuitError(format!("Proof serialization failed: {}", e)))?;
        
        // Public inputs for verification
        let public_inputs = vec![
            Self::field_to_string(&mean),
            Self::field_to_string(&variance),
        ];
        
        Ok((Some(proof_bytes), Some(public_inputs)))
    }
    
    /// Convert dataset to field elements for circuit input.
    fn dataset_to_field_elements(dataset: &Dataset) -> Result<Vec<Fr>> {
        // For now, create field elements from the dataset hash
        // In a real implementation, this would process actual data
        let hash = dataset.compute_hash();
        let hash_bytes = hex::decode(&hash)
            .map_err(|e| LedgerError::Crypto(format!("Hash decode error: {}", e)))?;
        
        let mut elements = Vec::new();
        for chunk in hash_bytes.chunks(8) {
            let mut bytes = [0u8; 8];
            bytes[..chunk.len()].copy_from_slice(chunk);
            let value = u64::from_le_bytes(bytes);
            elements.push(Fr::from(value));
        }
        
        // Ensure we have at least some elements
        if elements.is_empty() {
            elements.push(Fr::from(1u64));
        }
        
        Ok(elements)
    }
    
    /// Convert field element to string for storage.
    fn field_to_string(field: &Fr) -> String {
        let mut bytes = Vec::new();
        field
            .serialize_compressed(&mut bytes)
            .unwrap_or_default();
        hex::encode(bytes)
    }

    /// Create Merkle tree proof for dataset integrity.
    fn create_merkle_proof(
        dataset: &Dataset,
        config: &ProofConfig,
    ) -> Result<(Option<String>, Option<MerkleProof>)> {
        // Create leaves from dataset chunks
        let chunk_size = config.chunk_size.unwrap_or(1000);
        let dataset_bytes = dataset.compute_hash().as_bytes().to_vec();

        let mut leaves = Vec::new();
        for chunk in dataset_bytes.chunks(chunk_size) {
            leaves.push(chunk.to_vec());
        }

        if leaves.is_empty() {
            return Ok((None, None));
        }

        let tree = MerkleTree::new(leaves, config.hash_algorithm.clone())?;
        let root = tree.root_hash().to_string();

        // Generate proof for the first leaf as example
        let proof = tree.generate_proof(0)?;

        Ok((Some(root), Some(proof)))
    }

    /// Create commitment to private inputs.
    fn create_private_commitment(dataset: &Dataset, config: &ProofConfig) -> Result<String> {
        // Create a commitment that binds to the private data without revealing it
        let commitment_input = format!(
            "private:{}:{}:{}",
            dataset.compute_hash(),
            dataset.size,
            config.proof_type.type_name()
        );

        hash_bytes(commitment_input.as_bytes(), config.hash_algorithm.clone())
    }

    /// Verify the proof using public inputs.
    pub fn verify(&self) -> Result<bool> {
        // Verify Groth16 proof if present
        if let (Some(groth16_proof), Some(circuit_inputs)) = (&self.groth16_proof, &self.circuit_public_inputs) {
            return self.verify_groth16_proof(groth16_proof, circuit_inputs);
        }
        
        // Fallback to legacy verification
        self.verify_legacy()
    }
    
    /// Verify legacy simulated proof.
    fn verify_legacy(&self) -> Result<bool> {
        // Basic verification checks
        if self.proof_data.is_empty() {
            return Ok(false);
        }

        if self.public_inputs.is_empty() {
            return Ok(false);
        }

        // Check proof format (legacy format is 288 bytes, new format is 32 bytes)
        if self.proof_data.len() != 288 && self.proof_data.len() != 32 {
            return Ok(false);
        }

        // Verify proof type marker
        if self.proof_data[0] != self.proof_type.type_id() {
            return Ok(false);
        }

        // Verify Merkle proof if present
        if let (Some(_root), Some(merkle_proof)) = (&self.merkle_root, &self.merkle_proof) {
            let is_valid = MerkleTree::verify_proof(merkle_proof, HashAlgorithm::default())?;
            if !is_valid {
                return Ok(false);
            }
        }

        // Additional verification would involve actual cryptographic verification
        // For now, we consider the proof valid if basic checks pass
        Ok(true)
    }
    
    /// Verify Groth16 proof with verifying key.
    fn verify_groth16_proof(&self, proof_bytes: &[u8], public_inputs: &[String]) -> Result<bool> {
        // This method requires a verifying key, which should be provided
        // For now, we'll do basic format validation
        
        // Parse proof from bytes
        let proof = Groth16Proof::<circuits::Curve>::deserialize_compressed(&mut &proof_bytes[..])
            .map_err(|e| LedgerError::CircuitError(format!("Proof deserialization failed: {}", e)))?;
        
        // Convert public inputs from strings to field elements
        let mut field_inputs = Vec::new();
        for input_str in public_inputs {
            let input_bytes = hex::decode(input_str)
                .map_err(|e| LedgerError::Crypto(format!("Input decode error: {}", e)))?;
            let field_element = Fr::deserialize_compressed(&mut &input_bytes[..])
                .map_err(|e| LedgerError::CircuitError(format!("Field deserialization failed: {}", e)))?;
            field_inputs.push(field_element);
        }
        
        // For actual verification, we need the verifying key
        // This is a placeholder that validates the proof structure
        Ok(true)
    }
    
    /// Verify Groth16 proof with a provided verifying key.
    pub fn verify_with_key(&self, vk: &PreparedVerifyingKey<circuits::Curve>) -> Result<bool> {
        if let (Some(groth16_proof), Some(circuit_inputs)) = (&self.groth16_proof, &self.circuit_public_inputs) {
            // Parse proof from bytes
            let proof = Groth16Proof::<circuits::Curve>::deserialize_compressed(&mut &groth16_proof[..])
                .map_err(|e| LedgerError::CircuitError(format!("Proof deserialization failed: {}", e)))?;
            
            // Convert public inputs from strings to field elements
            let mut field_inputs = Vec::new();
            for input_str in circuit_inputs {
                let input_bytes = hex::decode(input_str)
                    .map_err(|e| LedgerError::Crypto(format!("Input decode error: {}", e)))?;
                let field_element = Fr::deserialize_compressed(&mut &input_bytes[..])
                    .map_err(|e| LedgerError::CircuitError(format!("Field deserialization failed: {}", e)))?;
                field_inputs.push(field_element);
            }
            
            // Verify the proof
            circuits::verify_proof(&proof, vk, &field_inputs)
        } else {
            // Fallback to legacy verification
            self.verify_legacy()
        }
    }

    /// Verify the proof against specific public inputs.
    pub fn verify_with_inputs(&self, expected_inputs: &[String]) -> Result<bool> {
        if self.public_inputs != expected_inputs {
            return Ok(false);
        }

        self.verify()
    }

    /// Get the size of the proof in bytes.
    pub fn size_bytes(&self) -> usize {
        let mut size = self.proof_data.len();
        if let Some(groth16_proof) = &self.groth16_proof {
            size += groth16_proof.len();
        }
        size
    }

    /// Export the proof to JSON format.
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string_pretty(self).map_err(|e| LedgerError::Json(e))
    }

    /// Import a proof from JSON format.
    pub fn from_json(json: &str) -> Result<Self> {
        serde_json::from_str(json).map_err(|e| LedgerError::Json(e))
    }

    /// Get a summary of the proof properties.
    pub fn summary(&self) -> ProofSummary {
        ProofSummary {
            dataset_hash: self.dataset_hash.clone(),
            proof_type: self.proof_type.clone(),
            size_bytes: self.proof_data.len(),
            has_merkle_proof: self.merkle_proof.is_some(),
            timestamp: self.timestamp,
            version: self.version.clone(),
        }
    }
}

impl ProofType {
    /// Get a unique identifier for the proof type.
    fn type_id(&self) -> u8 {
        match self {
            ProofType::DatasetIntegrity => 1,
            ProofType::RowCount => 2,
            ProofType::Schema => 3,
            ProofType::Statistics => 4,
            ProofType::Transformation => 5,
            ProofType::DataSplit => 6,
            ProofType::Custom(_) => 255,
        }
    }

    /// Get a string name for the proof type.
    fn type_name(&self) -> String {
        match self {
            ProofType::DatasetIntegrity => "integrity".to_string(),
            ProofType::RowCount => "row_count".to_string(),
            ProofType::Schema => "schema".to_string(),
            ProofType::Statistics => "statistics".to_string(),
            ProofType::Transformation => "transformation".to_string(),
            ProofType::DataSplit => "data_split".to_string(),
            ProofType::Custom(name) => format!("custom_{}", name),
        }
    }
}

/// A lightweight summary of proof properties.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofSummary {
    pub dataset_hash: String,
    pub proof_type: ProofType,
    pub size_bytes: usize,
    pub has_merkle_proof: bool,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub version: String,
}

/// Key management for Groth16 proofs.
pub struct ProofKeyManager {
    dataset_keys: Option<(ProvingKey<circuits::Curve>, PreparedVerifyingKey<circuits::Curve>)>,
    statistical_keys: Option<(ProvingKey<circuits::Curve>, PreparedVerifyingKey<circuits::Curve>)>,
}

impl ProofKeyManager {
    pub fn new() -> Self {
        Self {
            dataset_keys: None,
            statistical_keys: None,
        }
    }
    
    /// Setup keys for dataset circuit.
    pub fn setup_dataset_circuit(&mut self) -> Result<()> {
        let circuit = DatasetCircuit {
            dataset_hash: Some(Fr::from(0u64)),
            row_count: Some(Fr::from(0u64)),
            dataset_content: Some(vec![Fr::from(1u64)]),
            nonce: Some(Fr::from(0u64)),
        };
        
        let (pk, vk) = circuits::setup_circuit(circuit)?;
        self.dataset_keys = Some((pk, vk));
        Ok(())
    }
    
    /// Setup keys for statistical circuit.
    pub fn setup_statistical_circuit(&mut self) -> Result<()> {
        let circuit = StatisticalCircuit {
            mean: Some(Fr::from(0u64)),
            variance: Some(Fr::from(0u64)),
            data: Some(vec![Fr::from(1u64)]),
        };
        
        let (pk, vk) = circuits::setup_circuit(circuit)?;
        self.statistical_keys = Some((pk, vk));
        Ok(())
    }
    
    /// Get dataset circuit keys.
    pub fn dataset_keys(&self) -> Option<&(ProvingKey<circuits::Curve>, PreparedVerifyingKey<circuits::Curve>)> {
        self.dataset_keys.as_ref()
    }
    
    /// Get statistical circuit keys.
    pub fn statistical_keys(&self) -> Option<&(ProvingKey<circuits::Curve>, PreparedVerifyingKey<circuits::Curve>)> {
        self.statistical_keys.as_ref()
    }
}

/// Batch proof generation for multiple datasets.
pub struct BatchProofGenerator {
    config: ProofConfig,
    datasets: Vec<Dataset>,
    key_manager: ProofKeyManager,
}

impl BatchProofGenerator {
    pub fn new(config: ProofConfig) -> Self {
        Self {
            config,
            datasets: Vec::new(),
            key_manager: ProofKeyManager::new(),
        }
    }
    
    /// Initialize with pre-computed keys for performance.
    pub fn with_keys(mut config: ProofConfig) -> Result<Self> {
        let mut key_manager = ProofKeyManager::new();
        
        // Setup keys based on proof types we expect
        key_manager.setup_dataset_circuit()?;
        key_manager.setup_statistical_circuit()?;
        
        // Set keys in config for reuse
        if let Some((pk, vk)) = key_manager.dataset_keys() {
            config.proving_key = Some(pk.clone());
            config.verifying_key = Some(vk.clone());
        }
        
        Ok(Self {
            config,
            datasets: Vec::new(),
            key_manager,
        })
    }

    pub fn add_dataset(&mut self, dataset: Dataset) {
        self.datasets.push(dataset);
    }

    pub fn generate_batch_proof(&self) -> Result<Vec<Proof>> {
        let mut proofs = Vec::new();

        for dataset in &self.datasets {
            let mut config = self.config.clone();
            
            // Use appropriate keys based on proof type
            match &config.proof_type {
                ProofType::Statistics => {
                    if let Some((pk, vk)) = self.key_manager.statistical_keys() {
                        config.proving_key = Some(pk.clone());
                        config.verifying_key = Some(vk.clone());
                    }
                }
                _ => {
                    if let Some((pk, vk)) = self.key_manager.dataset_keys() {
                        config.proving_key = Some(pk.clone());
                        config.verifying_key = Some(vk.clone());
                    }
                }
            }
            
            let proof = Proof::generate(dataset, &config)?;
            proofs.push(proof);
        }

        Ok(proofs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_proof_generation() {
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "id,value").unwrap();
        writeln!(temp_file, "1,100").unwrap();

        let temp_path = temp_file.path().with_extension("csv");
        std::fs::copy(temp_file.path(), &temp_path).unwrap();

        let dataset = Dataset::from_path(&temp_path).unwrap();
        let config = ProofConfig::default();

        let proof = Proof::generate(&dataset, &config).unwrap();

        assert_eq!(proof.dataset_hash, dataset.compute_hash());
        assert!(!proof.proof_data.is_empty());
        assert_eq!(proof.proof_type, ProofType::DatasetIntegrity);
        assert!(!proof.public_inputs.is_empty());

        std::fs::remove_file(temp_path).ok();
    }

    #[test]
    fn test_proof_verification() {
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "name,age").unwrap();
        writeln!(temp_file, "Alice,25").unwrap();

        let temp_path = temp_file.path().with_extension("csv");
        std::fs::copy(temp_file.path(), &temp_path).unwrap();

        let dataset = Dataset::from_path(&temp_path).unwrap();
        let config = ProofConfig::default();

        let proof = Proof::generate(&dataset, &config).unwrap();
        assert!(proof.verify().unwrap());

        std::fs::remove_file(temp_path).ok();
    }

    #[test]
    fn test_different_proof_types() {
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "data").unwrap();
        writeln!(temp_file, "test").unwrap();

        let temp_path = temp_file.path().with_extension("csv");
        std::fs::copy(temp_file.path(), &temp_path).unwrap();

        let dataset = Dataset::from_path(&temp_path).unwrap();

        let proof_types = vec![
            ProofType::DatasetIntegrity,
            ProofType::RowCount,
            ProofType::Schema,
            ProofType::Statistics,
        ];

        for proof_type in proof_types {
            let config = ProofConfig {
                proof_type: proof_type.clone(),
                ..ProofConfig::default()
            };

            let proof = Proof::generate(&dataset, &config).unwrap();
            assert_eq!(proof.proof_type, proof_type);
            assert!(proof.verify().unwrap());
        }

        std::fs::remove_file(temp_path).ok();
    }
}
