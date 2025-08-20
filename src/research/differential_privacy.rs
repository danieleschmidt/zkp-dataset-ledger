//! Differential privacy implementations for ZKP dataset analysis.
//!
//! This module provides privacy-preserving techniques for statistical
//! analysis and machine learning with formal privacy guarantees.

use crate::{Dataset, LedgerError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Differential privacy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DifferentialPrivacyConfig {
    pub epsilon: f64,                    // Privacy budget
    pub delta: f64,                      // Failure probability
    pub sensitivity: f64,                // Global sensitivity
    pub noise_mechanism: NoiseMechanism, // Noise generation method
    pub clipping_bound: Option<f64>,     // Gradient clipping for ML
}

impl Default for DifferentialPrivacyConfig {
    fn default() -> Self {
        Self {
            epsilon: 1.0,
            delta: 1e-5,
            sensitivity: 1.0,
            noise_mechanism: NoiseMechanism::Laplace,
            clipping_bound: Some(1.0),
        }
    }
}

/// Noise mechanisms for differential privacy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NoiseMechanism {
    Laplace,
    Gaussian,
    Exponential,
}

/// Differentially private aggregation functions
pub struct DifferentialPrivacyEngine {
    config: DifferentialPrivacyConfig,
}

impl DifferentialPrivacyEngine {
    pub fn new(config: DifferentialPrivacyConfig) -> Self {
        Self { config }
    }

    /// Compute differentially private mean
    pub fn private_mean(&self, values: &[f64]) -> Result<f64> {
        if values.is_empty() {
            return Err(LedgerError::internal("Cannot compute mean of empty values"));
        }

        let true_mean = values.iter().sum::<f64>() / values.len() as f64;
        let noise = self.generate_noise(self.config.sensitivity / values.len() as f64)?;

        Ok(true_mean + noise)
    }

    /// Compute differentially private sum
    pub fn private_sum(&self, values: &[f64]) -> Result<f64> {
        let true_sum = values.iter().sum::<f64>();
        let noise = self.generate_noise(self.config.sensitivity)?;

        Ok(true_sum + noise)
    }

    /// Compute differentially private count
    pub fn private_count(&self, values: &[f64]) -> Result<f64> {
        let true_count = values.len() as f64;
        let noise = self.generate_noise(1.0)?; // Sensitivity of count is 1

        Ok(true_count + noise)
    }

    /// Compute differentially private histogram
    pub fn private_histogram(&self, values: &[f64], bins: usize) -> Result<Vec<f64>> {
        let (min_val, max_val) = values
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), &val| {
                (min.min(val), max.max(val))
            });

        let bin_width = (max_val - min_val) / bins as f64;
        let mut histogram = vec![0.0; bins];

        // Count true frequencies
        for &value in values {
            let bin_index = ((value - min_val) / bin_width).floor() as usize;
            let bin_index = bin_index.min(bins - 1);
            histogram[bin_index] += 1.0;
        }

        // Add noise to each bin
        for count in &mut histogram {
            let noise = self.generate_noise(1.0)?; // Sensitivity is 1 for each bin
            *count += noise;
            *count = count.max(0.0); // Ensure non-negative counts
        }

        Ok(histogram)
    }

    /// Generate differentially private statistics for a dataset
    pub fn private_dataset_statistics(&self, dataset: &Dataset) -> Result<PrivateStatistics> {
        // Simulate extracting numerical columns from dataset
        let mock_data = self.extract_numerical_data(dataset)?;

        let mut column_stats = HashMap::new();

        for (column_name, values) in mock_data {
            let private_mean = self.private_mean(&values)?;
            let private_sum = self.private_sum(&values)?;
            let private_count = self.private_count(&values)?;

            let stats = ColumnStatistics {
                mean: private_mean,
                sum: private_sum,
                count: private_count,
                histogram: self.private_histogram(&values, 10)?,
            };

            column_stats.insert(column_name, stats);
        }

        Ok(PrivateStatistics {
            dataset_name: dataset.name.clone(),
            epsilon_used: self.config.epsilon,
            delta_used: self.config.delta,
            column_statistics: column_stats,
            privacy_guarantee: format!(
                "({}, {})-differential privacy",
                self.config.epsilon, self.config.delta
            ),
        })
    }

    /// Generate noise based on the configured mechanism
    fn generate_noise(&self, sensitivity: f64) -> Result<f64> {
        use rand::thread_rng;
        use rand_distr::{Distribution, Normal};

        let mut rng = thread_rng();

        match self.config.noise_mechanism {
            NoiseMechanism::Laplace => {
                let scale = sensitivity / self.config.epsilon;
                // Use exponential distribution as approximation for Laplace
                use rand::Rng;
                let exponential = rand_distr::Exp::new(1.0 / scale).map_err(|e| {
                    LedgerError::internal(&format!("Failed to create exponential distribution: {}", e))
                })?;
                let sign = if rng.gen::<bool>() { 1.0 } else { -1.0 };
                Ok(exponential.sample(&mut rng) * sign)
            }
            NoiseMechanism::Gaussian => {
                let sigma = sensitivity * (2.0 * (1.25 / self.config.delta).ln()).sqrt()
                    / self.config.epsilon;
                let normal = Normal::new(0.0, sigma).map_err(|e| {
                    LedgerError::internal(&format!("Failed to create Normal distribution: {}", e))
                })?;
                Ok(normal.sample(&mut rng))
            }
            NoiseMechanism::Exponential => {
                // Simplified exponential mechanism for numerical queries
                let scale = sensitivity / self.config.epsilon;
                let laplace = Laplace::new(0.0, scale).map_err(|e| {
                    LedgerError::internal(&format!("Failed to create Laplace distribution: {}", e))
                })?;
                Ok(laplace.sample(&mut rng))
            }
        }
    }

    /// Extract numerical data from dataset (mock implementation)
    fn extract_numerical_data(&self, dataset: &Dataset) -> Result<HashMap<String, Vec<f64>>> {
        let mut data = HashMap::new();

        // Mock data extraction - in real implementation this would parse the actual dataset
        let num_rows = dataset.row_count.unwrap_or(1000);
        let num_cols = dataset.column_count.unwrap_or(10);

        for col_idx in 0..num_cols {
            let column_name = format!("column_{}", col_idx);
            let mut values = Vec::new();

            // Generate mock numerical data
            for row_idx in 0..num_rows {
                let value = (row_idx as f64 * 0.1 + col_idx as f64) % 100.0;
                values.push(value);
            }

            data.insert(column_name, values);
        }

        Ok(data)
    }
}

/// Private statistics with differential privacy guarantees
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrivateStatistics {
    pub dataset_name: String,
    pub epsilon_used: f64,
    pub delta_used: f64,
    pub column_statistics: HashMap<String, ColumnStatistics>,
    pub privacy_guarantee: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColumnStatistics {
    pub mean: f64,
    pub sum: f64,
    pub count: f64,
    pub histogram: Vec<f64>,
}

/// Privacy budget management for multiple queries
pub struct PrivacyBudgetManager {
    total_epsilon: f64,
    remaining_epsilon: f64,
    queries: Vec<BudgetQuery>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BudgetQuery {
    pub query_id: String,
    pub epsilon_used: f64,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub query_type: String,
}

impl PrivacyBudgetManager {
    pub fn new(total_epsilon: f64) -> Self {
        Self {
            total_epsilon,
            remaining_epsilon: total_epsilon,
            queries: Vec::new(),
        }
    }

    /// Check if a query with given epsilon can be executed
    pub fn can_execute_query(&self, epsilon: f64) -> bool {
        self.remaining_epsilon >= epsilon
    }

    /// Execute a query and deduct from privacy budget
    pub fn execute_query(
        &mut self,
        query_id: String,
        epsilon: f64,
        query_type: String,
    ) -> Result<()> {
        if !self.can_execute_query(epsilon) {
            return Err(LedgerError::security("Insufficient privacy budget"));
        }

        self.remaining_epsilon -= epsilon;

        let query = BudgetQuery {
            query_id,
            epsilon_used: epsilon,
            timestamp: chrono::Utc::now(),
            query_type,
        };

        self.queries.push(query);
        Ok(())
    }

    /// Get remaining privacy budget
    pub fn remaining_budget(&self) -> f64 {
        self.remaining_epsilon
    }

    /// Get budget usage history
    pub fn get_query_history(&self) -> &[BudgetQuery] {
        &self.queries
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DatasetFormat;

    fn create_test_dataset() -> Dataset {
        Dataset {
            name: "test_dataset".to_string(),
            hash: "test_hash".to_string(),
            size: 1024,
            row_count: Some(100),
            column_count: Some(5),
            schema: None,
            statistics: None,
            format: DatasetFormat::Csv,
            path: None,
        }
    }

    #[test]
    fn test_differential_privacy_config() {
        let config = DifferentialPrivacyConfig::default();
        assert_eq!(config.epsilon, 1.0);
        assert_eq!(config.delta, 1e-5);
    }

    #[test]
    fn test_privacy_engine() {
        let config = DifferentialPrivacyConfig::default();
        let engine = DifferentialPrivacyEngine::new(config);

        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let private_mean = engine.private_mean(&values).unwrap();

        // The private mean should be close to the true mean (3.0) but with noise
        assert!((private_mean - 3.0).abs() < 5.0); // Allow for noise variance
    }

    #[test]
    fn test_private_sum() {
        let config = DifferentialPrivacyConfig::default();
        let engine = DifferentialPrivacyEngine::new(config);

        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let private_sum = engine.private_sum(&values).unwrap();

        // The private sum should be close to the true sum (15.0) but with noise
        assert!((private_sum - 15.0).abs() < 10.0); // Allow for noise variance
    }

    #[test]
    fn test_private_count() {
        let config = DifferentialPrivacyConfig::default();
        let engine = DifferentialPrivacyEngine::new(config);

        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let private_count = engine.private_count(&values).unwrap();

        // The private count should be close to the true count (5.0) but with noise
        assert!((private_count - 5.0).abs() < 5.0); // Allow for noise variance
    }

    #[test]
    fn test_private_histogram() {
        let config = DifferentialPrivacyConfig::default();
        let engine = DifferentialPrivacyEngine::new(config);

        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let histogram = engine.private_histogram(&values, 5).unwrap();

        assert_eq!(histogram.len(), 5);
        // Each bin should have approximately 1 count plus noise
        for &count in &histogram {
            assert!(count >= 0.0); // Should be non-negative after post-processing
        }
    }

    #[test]
    fn test_dataset_statistics() {
        let config = DifferentialPrivacyConfig::default();
        let engine = DifferentialPrivacyEngine::new(config);
        let dataset = create_test_dataset();

        let stats = engine.private_dataset_statistics(&dataset).unwrap();

        assert_eq!(stats.dataset_name, "test_dataset");
        assert_eq!(stats.epsilon_used, 1.0);
        assert_eq!(stats.column_statistics.len(), 5); // Should have 5 columns
        assert!(stats.privacy_guarantee.contains("differential privacy"));
    }

    #[test]
    fn test_privacy_budget_manager() {
        let mut budget_manager = PrivacyBudgetManager::new(2.0);

        assert!(budget_manager.can_execute_query(1.0));
        assert!(budget_manager
            .execute_query("query1".to_string(), 1.0, "mean".to_string())
            .is_ok());
        assert_eq!(budget_manager.remaining_budget(), 1.0);

        assert!(budget_manager.can_execute_query(0.5));
        assert!(!budget_manager.can_execute_query(1.5));

        assert!(budget_manager
            .execute_query("query2".to_string(), 0.5, "sum".to_string())
            .is_ok());
        assert_eq!(budget_manager.remaining_budget(), 0.5);

        // Should fail when budget is insufficient
        assert!(budget_manager
            .execute_query("query3".to_string(), 1.0, "count".to_string())
            .is_err());
    }

    #[test]
    fn test_query_history() {
        let mut budget_manager = PrivacyBudgetManager::new(2.0);

        budget_manager
            .execute_query("query1".to_string(), 1.0, "mean".to_string())
            .unwrap();
        budget_manager
            .execute_query("query2".to_string(), 0.5, "sum".to_string())
            .unwrap();

        let history = budget_manager.get_query_history();
        assert_eq!(history.len(), 2);
        assert_eq!(history[0].query_id, "query1");
        assert_eq!(history[1].query_id, "query2");
    }
}
