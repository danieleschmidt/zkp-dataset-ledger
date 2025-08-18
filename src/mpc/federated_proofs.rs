//! Multi-Party Computation for Federated Zero-Knowledge Proofs
//!
//! This module enables multiple parties to collaboratively generate ZK proofs
//! about their combined datasets without revealing individual data.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use tokio::sync::{broadcast, mpsc, RwLock as AsyncRwLock};
use uuid::Uuid;

use crate::circuits::{DatasetProperty, ZKProof};
use crate::{LedgerError, Result};

/// Configuration for federated proof generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederatedConfig {
    pub min_participants: usize,
    pub max_participants: usize,
    pub aggregation_method: AggregationMethod,
    pub privacy_threshold: f64,
    pub timeout_seconds: u64,
    pub require_unanimous: bool,
}

impl Default for FederatedConfig {
    fn default() -> Self {
        Self {
            min_participants: 2,
            max_participants: 10,
            aggregation_method: AggregationMethod::SecureSum,
            privacy_threshold: 0.95,
            timeout_seconds: 300, // 5 minutes
            require_unanimous: false,
        }
    }
}

/// Methods for aggregating federated proofs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AggregationMethod {
    /// Simple secure summation
    SecureSum,
    /// Threshold aggregation (k-out-of-n)
    Threshold { threshold: usize },
    /// Weighted aggregation by data size
    Weighted,
    /// Secure multiparty computation
    SecureMPC,
    /// Differential privacy aggregation
    DifferentialPrivate { epsilon: f64, delta: f64 },
}

/// Participant in federated proof generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederatedParticipant {
    pub participant_id: String,
    pub public_key: String,
    pub endpoint: String,
    pub reputation_score: f64,
    pub data_contribution: u64,
    pub last_activity: DateTime<Utc>,
}

/// Status of a participant in the federation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ParticipantStatus {
    Pending,
    Active,
    Contributing,
    Completed,
    Failed { error: String },
    Timeout,
}

/// A federated proof session
#[derive(Debug)]
pub struct FederatedProofSession {
    pub session_id: String,
    pub coordinator_id: String,
    pub participants: HashMap<String, FederatedParticipant>,
    pub participant_statuses: HashMap<String, ParticipantStatus>,
    pub partial_proofs: HashMap<String, PartialProof>,
    pub config: FederatedConfig,
    pub target_properties: Vec<DatasetProperty>,
    pub created_at: DateTime<Utc>,
    pub deadline: DateTime<Utc>,
    pub aggregated_proof: Option<ZKProof>,
    pub state: SessionState,

    // Communication channels
    broadcast_tx: broadcast::Sender<FederatedMessage>,
    command_tx: mpsc::Sender<SessionCommand>,
    metrics: AsyncRwLock<FederatedMetrics>,
}

/// State of a federated proof session
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SessionState {
    Initializing,
    WaitingForParticipants,
    ProofGeneration,
    Aggregating,
    Completed,
    Failed { reason: String },
    Cancelled,
}

/// Partial proof from a participant
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartialProof {
    pub participant_id: String,
    pub proof_data: Vec<u8>,
    pub public_commitment: Vec<u8>,
    pub contribution_weight: f64,
    pub verification_key: String,
    pub timestamp: DateTime<Utc>,
    pub metadata: HashMap<String, String>,
}

/// Messages exchanged in federated protocol
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FederatedMessage {
    /// Invitation to join proof session
    JoinInvitation {
        session_id: String,
        coordinator_id: String,
        target_properties: Vec<DatasetProperty>,
        deadline: DateTime<Utc>,
    },
    /// Response to join invitation
    JoinResponse {
        participant_id: String,
        accepted: bool,
        public_key: String,
        estimated_contribution: u64,
    },
    /// Start proof generation phase
    StartProofGeneration {
        session_id: String,
        participants: Vec<String>,
        aggregation_params: AggregationParameters,
    },
    /// Submit partial proof
    PartialProofSubmission {
        participant_id: String,
        partial_proof: PartialProof,
    },
    /// Aggregation progress update
    AggregationUpdate {
        completed_participants: usize,
        total_participants: usize,
        estimated_completion: DateTime<Utc>,
    },
    /// Final aggregated proof result
    FinalProof {
        session_id: String,
        aggregated_proof: ZKProof,
        participant_contributions: HashMap<String, f64>,
    },
    /// Error or status update
    StatusUpdate {
        participant_id: String,
        status: ParticipantStatus,
        message: String,
    },
}

/// Parameters for proof aggregation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregationParameters {
    pub method: AggregationMethod,
    pub security_parameter: u32,
    pub privacy_budget: Option<f64>,
    pub threshold_params: HashMap<String, f64>,
}

/// Internal commands for session management
#[derive(Debug)]
pub enum SessionCommand {
    AddParticipant(FederatedParticipant),
    RemoveParticipant(String),
    StartProofGeneration,
    SubmitPartialProof(PartialProof),
    CheckTimeout,
    Aggregate,
    Cancel(String),
}

/// Metrics for federated proof session
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederatedMetrics {
    pub total_participants: usize,
    pub active_participants: usize,
    pub completed_proofs: usize,
    pub failed_proofs: usize,
    pub total_data_processed: u64,
    pub aggregation_time_ms: u64,
    pub communication_rounds: u32,
    pub privacy_loss: f64,
    pub throughput_mbps: f64,
}

impl FederatedProofSession {
    /// Create a new federated proof session
    pub fn new(
        coordinator_id: String,
        target_properties: Vec<DatasetProperty>,
        config: FederatedConfig,
    ) -> Result<Self> {
        let session_id = format!("fed-{}", Uuid::new_v4());
        let created_at = Utc::now();
        let deadline = created_at + chrono::Duration::seconds(config.timeout_seconds as i64);

        let (broadcast_tx, _) = broadcast::channel(1000);
        let (command_tx, command_rx) = mpsc::channel(100);

        log::info!(
            "Creating federated proof session {} for {} properties",
            session_id,
            target_properties.len()
        );

        let mut session = Self {
            session_id: session_id.clone(),
            coordinator_id,
            participants: HashMap::new(),
            participant_statuses: HashMap::new(),
            partial_proofs: HashMap::new(),
            config,
            target_properties,
            created_at,
            deadline,
            aggregated_proof: None,
            state: SessionState::Initializing,
            broadcast_tx,
            command_tx,
            metrics: AsyncRwLock::new(FederatedMetrics::default()),
        };

        // Start background session management
        session.start_session_manager(command_rx)?;

        session.state = SessionState::WaitingForParticipants;
        Ok(session)
    }

    /// Add a participant to the federation
    pub async fn add_participant(&mut self, participant: FederatedParticipant) -> Result<()> {
        if self.participants.len() >= self.config.max_participants {
            return Err(LedgerError::invalid_input(
                "participant_count",
                "Maximum participants exceeded",
            ));
        }

        log::info!(
            "Adding participant {} to session {}",
            participant.participant_id,
            self.session_id
        );

        self.participant_statuses.insert(
            participant.participant_id.clone(),
            ParticipantStatus::Pending,
        );

        self.participants
            .insert(participant.participant_id.clone(), participant);

        // Update metrics
        let mut metrics = self.metrics.write().await;
        metrics.total_participants = self.participants.len();

        // Send join invitation
        let invitation = FederatedMessage::JoinInvitation {
            session_id: self.session_id.clone(),
            coordinator_id: self.coordinator_id.clone(),
            target_properties: self.target_properties.clone(),
            deadline: self.deadline,
        };

        self.broadcast_message(invitation).await?;

        Ok(())
    }

    /// Start the proof generation phase
    pub async fn start_proof_generation(&mut self) -> Result<()> {
        if self.participants.len() < self.config.min_participants {
            return Err(LedgerError::invalid_input(
                "participant_count",
                "Insufficient participants",
            ));
        }

        log::info!(
            "Starting proof generation for session {} with {} participants",
            self.session_id,
            self.participants.len()
        );

        self.state = SessionState::ProofGeneration;

        // Prepare aggregation parameters
        let aggregation_params = AggregationParameters {
            method: self.config.aggregation_method.clone(),
            security_parameter: 128,
            privacy_budget: Some(self.config.privacy_threshold),
            threshold_params: HashMap::new(),
        };

        // Notify all participants to start proof generation
        let start_message = FederatedMessage::StartProofGeneration {
            session_id: self.session_id.clone(),
            participants: self.participants.keys().cloned().collect(),
            aggregation_params,
        };

        self.broadcast_message(start_message).await?;

        // Update participant statuses
        for participant_id in self.participants.keys() {
            self.participant_statuses
                .insert(participant_id.clone(), ParticipantStatus::Contributing);
        }

        Ok(())
    }

    /// Submit a partial proof from a participant
    pub async fn submit_partial_proof(&mut self, partial_proof: PartialProof) -> Result<()> {
        if self.state != SessionState::ProofGeneration {
            return Err(LedgerError::invalid_input(
                "session_state",
                "Not in proof generation phase",
            ));
        }

        if !self
            .participants
            .contains_key(&partial_proof.participant_id)
        {
            return Err(LedgerError::invalid_input(
                "participant_id",
                "Unknown participant",
            ));
        }

        log::info!(
            "Received partial proof from participant {} for session {}",
            partial_proof.participant_id,
            self.session_id
        );

        // Validate partial proof
        self.validate_partial_proof(&partial_proof).await?;

        // Update participant status
        self.participant_statuses.insert(
            partial_proof.participant_id.clone(),
            ParticipantStatus::Completed,
        );

        // Store partial proof
        self.partial_proofs
            .insert(partial_proof.participant_id.clone(), partial_proof.clone());

        // Update metrics
        {
            let mut metrics = self.metrics.write().await;
            metrics.completed_proofs += 1;
            metrics.total_data_processed += partial_proof.contribution_weight as u64;
        }

        // Check if we have enough proofs to aggregate
        if self.can_aggregate().await {
            self.start_aggregation().await?;
        }

        // Broadcast progress update
        let update = FederatedMessage::AggregationUpdate {
            completed_participants: self.partial_proofs.len(),
            total_participants: self.participants.len(),
            estimated_completion: self.estimate_completion_time().await,
        };

        self.broadcast_message(update).await?;

        Ok(())
    }

    /// Aggregate partial proofs into final proof
    pub async fn aggregate_proofs(&mut self) -> Result<ZKProof> {
        let start_time = std::time::Instant::now();

        log::info!(
            "Starting proof aggregation for session {} with {} partial proofs",
            self.session_id,
            self.partial_proofs.len()
        );

        self.state = SessionState::Aggregating;

        // Perform aggregation based on method
        let aggregated_proof = match &self.config.aggregation_method {
            AggregationMethod::SecureSum => self.aggregate_secure_sum().await?,
            AggregationMethod::Threshold { threshold } => {
                self.aggregate_threshold(*threshold).await?
            }
            AggregationMethod::Weighted => self.aggregate_weighted().await?,
            AggregationMethod::SecureMPC => self.aggregate_secure_mpc().await?,
            AggregationMethod::DifferentialPrivate { epsilon, delta } => {
                self.aggregate_differential_private(*epsilon, *delta)
                    .await?
            }
        };

        let aggregation_time = start_time.elapsed().as_millis() as u64;

        // Update metrics
        let mut metrics = self.metrics.write().await;
        metrics.aggregation_time_ms = aggregation_time;
        metrics.communication_rounds += 1;

        self.aggregated_proof = Some(aggregated_proof.clone());
        self.state = SessionState::Completed;

        // Calculate participant contributions
        let contributions = self.calculate_contributions().await;

        // Broadcast final result
        let final_message = FederatedMessage::FinalProof {
            session_id: self.session_id.clone(),
            aggregated_proof: aggregated_proof.clone(),
            participant_contributions: contributions,
        };

        self.broadcast_message(final_message).await?;

        log::info!(
            "Completed proof aggregation for session {} in {}ms",
            self.session_id,
            aggregation_time
        );

        Ok(aggregated_proof)
    }

    /// Validate a partial proof submission
    async fn validate_partial_proof(&self, partial_proof: &PartialProof) -> Result<()> {
        // Check timestamp validity
        if partial_proof.timestamp > Utc::now() {
            return Err(LedgerError::invalid_input(
                "timestamp",
                "Future timestamp not allowed",
            ));
        }

        // Check if proof data is not empty
        if partial_proof.proof_data.is_empty() {
            return Err(LedgerError::invalid_input("proof_data", "Empty proof data"));
        }

        // Validate contribution weight
        if partial_proof.contribution_weight <= 0.0 || partial_proof.contribution_weight > 1.0 {
            return Err(LedgerError::invalid_input(
                "contribution_weight",
                "Invalid contribution weight",
            ));
        }

        // Validate proof format (basic check)
        if partial_proof.proof_data.len() < 32 {
            return Err(LedgerError::invalid_input(
                "proof_data",
                "Proof data too short",
            ));
        }

        Ok(())
    }

    /// Check if we can start aggregation
    async fn can_aggregate(&self) -> bool {
        let completed_count = self.partial_proofs.len();
        let total_count = self.participants.len();

        if self.config.require_unanimous {
            completed_count == total_count
        } else {
            completed_count >= self.config.min_participants
        }
    }

    /// Start the aggregation process
    async fn start_aggregation(&mut self) -> Result<()> {
        self.command_tx
            .send(SessionCommand::Aggregate)
            .await
            .map_err(|e| LedgerError::storage_error("command_send", e.to_string()))?;
        Ok(())
    }

    /// Estimate completion time based on current progress
    async fn estimate_completion_time(&self) -> DateTime<Utc> {
        let completed = self.partial_proofs.len() as f64;
        let total = self.participants.len() as f64;
        let progress = completed / total;

        if progress > 0.0 {
            let elapsed = Utc::now().signed_duration_since(self.created_at);
            let estimated_total = elapsed.num_seconds() as f64 / progress;
            Utc::now() + chrono::Duration::seconds(estimated_total as i64 - elapsed.num_seconds())
        } else {
            self.deadline
        }
    }

    /// Aggregate using secure summation
    async fn aggregate_secure_sum(&self) -> Result<ZKProof> {
        log::info!("Performing secure sum aggregation");

        let mut aggregated_data = Vec::new();
        let mut total_weight = 0.0;

        for partial_proof in self.partial_proofs.values() {
            // Simulate secure summation of proof components
            if aggregated_data.is_empty() {
                aggregated_data = partial_proof.proof_data.clone();
            } else {
                for (i, &byte) in partial_proof.proof_data.iter().enumerate() {
                    if i < aggregated_data.len() {
                        aggregated_data[i] = aggregated_data[i].wrapping_add(byte);
                    }
                }
            }
            total_weight += partial_proof.contribution_weight;
        }

        // Normalize by total weight
        if total_weight > 0.0 {
            for byte in &mut aggregated_data {
                *byte = ((*byte as f64) / total_weight) as u8;
            }
        }

        self.create_aggregated_proof(aggregated_data).await
    }

    /// Aggregate using threshold method
    async fn aggregate_threshold(&self, threshold: usize) -> Result<ZKProof> {
        log::info!(
            "Performing threshold aggregation with threshold {}",
            threshold
        );

        if self.partial_proofs.len() < threshold {
            return Err(LedgerError::invalid_input(
                "threshold",
                "Insufficient proofs for threshold",
            ));
        }

        // Select top proofs by contribution weight
        let mut sorted_proofs: Vec<_> = self.partial_proofs.values().collect();
        sorted_proofs.sort_by(|a, b| {
            b.contribution_weight
                .partial_cmp(&a.contribution_weight)
                .unwrap()
        });

        let selected_proofs = &sorted_proofs[..threshold];

        // Combine selected proofs
        let mut aggregated_data = Vec::new();
        for (i, proof) in selected_proofs.iter().enumerate() {
            if i == 0 {
                aggregated_data = proof.proof_data.clone();
            } else {
                for (j, &byte) in proof.proof_data.iter().enumerate() {
                    if j < aggregated_data.len() {
                        aggregated_data[j] ^= byte; // XOR combination
                    }
                }
            }
        }

        self.create_aggregated_proof(aggregated_data).await
    }

    /// Aggregate using weighted method
    async fn aggregate_weighted(&self) -> Result<ZKProof> {
        log::info!("Performing weighted aggregation");

        let mut weighted_data = vec![0.0; 64]; // Fixed size for simplicity
        let mut total_weight = 0.0;

        for partial_proof in self.partial_proofs.values() {
            let weight = partial_proof.contribution_weight;
            total_weight += weight;

            for (i, &byte) in partial_proof.proof_data.iter().take(64).enumerate() {
                weighted_data[i] += (byte as f64) * weight;
            }
        }

        // Normalize by total weight
        let aggregated_data: Vec<u8> = weighted_data
            .iter()
            .map(|&val| (val / total_weight) as u8)
            .collect();

        self.create_aggregated_proof(aggregated_data).await
    }

    /// Aggregate using secure MPC
    async fn aggregate_secure_mpc(&self) -> Result<ZKProof> {
        log::info!("Performing secure MPC aggregation");

        // Simulate secure multiparty computation
        let mut secret_shares = Vec::new();

        for partial_proof in self.partial_proofs.values() {
            // Split proof data into secret shares
            let shares = self
                .create_secret_shares(&partial_proof.proof_data, self.participants.len())
                .await?;
            secret_shares.push(shares);
        }

        // Reconstruct secrets and aggregate
        let mut aggregated_data = Vec::new();
        for i in 0..secret_shares[0].len() {
            let mut combined_share = 0u8;
            for shares in &secret_shares {
                combined_share ^= shares[i];
            }
            aggregated_data.push(combined_share);
        }

        self.create_aggregated_proof(aggregated_data).await
    }

    /// Aggregate with differential privacy
    async fn aggregate_differential_private(&self, epsilon: f64, delta: f64) -> Result<ZKProof> {
        log::info!(
            "Performing differential private aggregation (ε={}, δ={})",
            epsilon,
            delta
        );

        // First perform basic aggregation
        let base_aggregation = self.aggregate_secure_sum().await?;

        // Add calibrated noise for differential privacy
        let mut noisy_data = base_aggregation.proof_data.clone();
        let noise_scale = self.calculate_noise_scale(epsilon, delta);

        for byte in &mut noisy_data {
            let noise = self.sample_laplace_noise(noise_scale);
            *byte = (*byte as f64 + noise).max(0.0).min(255.0) as u8;
        }

        self.create_aggregated_proof(noisy_data).await
    }

    /// Create final aggregated proof
    async fn create_aggregated_proof(&self, proof_data: Vec<u8>) -> Result<ZKProof> {
        let proof_id = format!("fed-proof-{}-{}", self.session_id, Utc::now().timestamp());

        // Generate verification key for aggregated proof
        let mut hasher = Sha256::new();
        hasher.update(&proof_data);
        hasher.update(self.session_id.as_bytes());
        let verification_key = format!("{:x}", hasher.finalize());

        // Serialize public inputs (participant contributions)
        let public_inputs = serde_json::to_vec(&self.calculate_contributions().await)?;

        let proof_size = proof_data.len();

        Ok(ZKProof {
            proof_id,
            circuit_id: format!("federated-{}", self.session_id),
            proof_data,
            public_inputs,
            verification_key,
            timestamp: Utc::now(),
            proof_size,
            generation_time_ms: {
                let metrics = self.metrics.read().await;
                metrics.aggregation_time_ms
            },
        })
    }

    /// Calculate participant contributions
    async fn calculate_contributions(&self) -> HashMap<String, f64> {
        let mut contributions = HashMap::new();
        let total_weight: f64 = self
            .partial_proofs
            .values()
            .map(|p| p.contribution_weight)
            .sum();

        for (participant_id, partial_proof) in &self.partial_proofs {
            let contribution = if total_weight > 0.0 {
                partial_proof.contribution_weight / total_weight
            } else {
                1.0 / self.partial_proofs.len() as f64
            };
            contributions.insert(participant_id.clone(), contribution);
        }

        contributions
    }

    /// Create secret shares for MPC
    async fn create_secret_shares(&self, data: &[u8], num_shares: usize) -> Result<Vec<u8>> {
        // Simple secret sharing scheme (in practice, use Shamir's secret sharing)
        let mut shares = Vec::new();

        for &byte in data {
            let mut share_sum = 0u8;
            for _ in 0..num_shares - 1 {
                let random_share = (Utc::now().timestamp_nanos_opt().unwrap_or(0) % 256) as u8;
                share_sum ^= random_share;
                shares.push(random_share);
            }
            shares.push(byte ^ share_sum); // Final share to reconstruct
        }

        Ok(shares)
    }

    /// Calculate noise scale for differential privacy
    fn calculate_noise_scale(&self, epsilon: f64, delta: f64) -> f64 {
        // Simplified noise calculation (in practice, use proper DP mechanisms)
        let sensitivity = 1.0; // Global sensitivity
        let noise_scale = sensitivity / epsilon;

        // Adjust for delta if using (ε,δ)-DP
        if delta > 0.0 {
            noise_scale * (1.0 + (1.0 / delta).ln()).sqrt()
        } else {
            noise_scale
        }
    }

    /// Sample from Laplace distribution for DP noise
    fn sample_laplace_noise(&self, scale: f64) -> f64 {
        // Simple Laplace noise (in practice, use proper random number generation)
        let uniform = (Utc::now().timestamp_nanos_opt().unwrap_or(0) % 1000000) as f64 / 1000000.0;
        let sign = if uniform < 0.5 { -1.0 } else { 1.0 };
        sign * scale * (2.0 * (uniform - 0.5).abs()).ln()
    }

    /// Broadcast message to all participants
    async fn broadcast_message(&self, message: FederatedMessage) -> Result<()> {
        self.broadcast_tx
            .send(message)
            .map_err(|e| LedgerError::storage_error("broadcast_send", e.to_string()))?;
        Ok(())
    }

    /// Start background session management
    fn start_session_manager(&self, mut command_rx: mpsc::Receiver<SessionCommand>) -> Result<()> {
        let session_id = self.session_id.clone();

        tokio::spawn(async move {
            log::info!("Starting session manager for {}", session_id);

            while let Some(command) = command_rx.recv().await {
                match command {
                    SessionCommand::CheckTimeout => {
                        // Handle timeout checks
                        log::debug!("Checking timeout for session {}", session_id);
                    }
                    SessionCommand::Aggregate => {
                        // Handle aggregation trigger
                        log::info!("Aggregation triggered for session {}", session_id);
                    }
                    SessionCommand::Cancel(reason) => {
                        log::warn!("Session {} cancelled: {}", session_id, reason);
                        break;
                    }
                    _ => {
                        // Handle other commands
                        log::debug!("Processing command for session {}", session_id);
                    }
                }
            }

            log::info!("Session manager for {} stopped", session_id);
        });

        Ok(())
    }

    /// Get current session metrics
    pub async fn get_metrics(&self) -> FederatedMetrics {
        self.metrics.read().await.clone()
    }
}

impl Default for FederatedMetrics {
    fn default() -> Self {
        Self {
            total_participants: 0,
            active_participants: 0,
            completed_proofs: 0,
            failed_proofs: 0,
            total_data_processed: 0,
            aggregation_time_ms: 0,
            communication_rounds: 0,
            privacy_loss: 0.0,
            throughput_mbps: 0.0,
        }
    }
}

/// Factory for creating federated proof sessions
pub struct FederatedProofFactory;

impl FederatedProofFactory {
    /// Create a new federated proof session
    pub fn create_session(
        coordinator_id: String,
        target_properties: Vec<DatasetProperty>,
        config: Option<FederatedConfig>,
    ) -> Result<FederatedProofSession> {
        let config = config.unwrap_or_default();
        FederatedProofSession::new(coordinator_id, target_properties, config)
    }

    /// Create a session for privacy-preserving statistics
    pub fn create_privacy_statistics_session(
        coordinator_id: String,
        epsilon: f64,
        delta: f64,
    ) -> Result<FederatedProofSession> {
        let properties = vec![
            DatasetProperty::Statistics {
                mean_range: (0.0, 100.0),
                variance_range: (0.0, 10000.0),
                distribution_type: crate::circuits::DistributionType::Normal,
            },
            DatasetProperty::Privacy {
                anonymization_level: 5,
                k_anonymity: Some(5),
                l_diversity: Some(2),
            },
        ];

        let config = FederatedConfig {
            aggregation_method: AggregationMethod::DifferentialPrivate { epsilon, delta },
            privacy_threshold: 0.99,
            ..Default::default()
        };

        FederatedProofSession::new(coordinator_id, properties, config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_federated_session_creation() {
        let properties = vec![
            DatasetProperty::RowCount { count: 1000 },
            DatasetProperty::DataQuality {
                completeness_ratio: 0.95,
                uniqueness_ratio: 0.99,
                validity_score: 0.98,
            },
        ];

        let session = FederatedProofSession::new(
            "coordinator-1".to_string(),
            properties,
            FederatedConfig::default(),
        )
        .unwrap();

        assert_eq!(session.participants.len(), 0);
        assert!(matches!(
            session.state,
            SessionState::WaitingForParticipants
        ));
    }

    #[tokio::test]
    async fn test_participant_management() {
        let properties = vec![DatasetProperty::RowCount { count: 100 }];
        let mut session = FederatedProofSession::new(
            "coordinator-1".to_string(),
            properties,
            FederatedConfig::default(),
        )
        .unwrap();

        let participant = FederatedParticipant {
            participant_id: "participant-1".to_string(),
            public_key: "key123".to_string(),
            endpoint: "http://participant1.com".to_string(),
            reputation_score: 0.95,
            data_contribution: 1000,
            last_activity: Utc::now(),
        };

        session.add_participant(participant).await.unwrap();
        assert_eq!(session.participants.len(), 1);
    }
}
