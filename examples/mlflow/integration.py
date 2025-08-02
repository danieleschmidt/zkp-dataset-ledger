#!/usr/bin/env python3
"""
MLflow Integration Example for ZKP Dataset Ledger

This example demonstrates how to integrate ZKP dataset proofs
with MLflow experiment tracking for complete ML pipeline audibility.

Features demonstrated:
- Automatic dataset proof generation during MLflow runs
- Model metadata with dataset provenance
- Proof verification during model loading
- Compliance report generation
"""

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import json
from pathlib import Path

# Import Python bindings (would be installed via pip install zkp-dataset-ledger)
try:
    from zkp_dataset_ledger import Ledger, ProofConfig
except ImportError:
    print("⚠️  Python bindings not available. Install with: pip install zkp-dataset-ledger")
    exit(1)


class MLflowZKPIntegration:
    """Integration class for MLflow + ZKP Dataset Ledger"""
    
    def __init__(self, project_name: str):
        self.ledger = Ledger(project_name)
        self.project_name = project_name
    
    def log_dataset_with_proof(self, 
                              data_path: str, 
                              dataset_name: str,
                              prove_properties: dict = None) -> str:
        """Log dataset to MLflow with ZKP proof"""
        
        # Generate ZKP proof
        proof_config = ProofConfig(
            prove_row_count=prove_properties.get('row_count', True),
            prove_schema=prove_properties.get('schema', True),
            prove_statistics=prove_properties.get('statistics', False),
            privacy_level='high' if prove_properties.get('private_columns') else 'standard'
        )
        
        proof = self.ledger.notarize_dataset_file(
            data_path, 
            dataset_name, 
            proof_config
        )
        
        # Log to MLflow
        mlflow.log_artifact(data_path, "datasets")
        mlflow.log_param(f"dataset_{dataset_name}_hash", proof.dataset_hash())
        mlflow.log_param(f"dataset_{dataset_name}_proof_size", proof.size_bytes())
        
        # Store proof as artifact
        proof_path = f"proof_{dataset_name}.json"
        with open(proof_path, 'w') as f:
            json.dump(proof.to_dict(), f, indent=2)
        mlflow.log_artifact(proof_path, "proofs")
        
        # Log dataset metadata
        mlflow.log_dict({
            "dataset_name": dataset_name,
            "proof_id": proof.id(),
            "timestamp": proof.timestamp().isoformat(),
            "verification_status": "verified" if self.ledger.verify_proof(proof) else "failed"
        }, f"dataset_{dataset_name}_metadata.json")
        
        return proof.id()
    
    def log_transformation(self, 
                          input_dataset: str,
                          output_dataset: str, 
                          operations: list,
                          output_path: str) -> str:
        """Log data transformation with proof"""
        
        transform_proof = self.ledger.record_transformation_file(
            input_dataset,
            output_dataset,
            operations,
            output_path
        )
        
        # Log transformation details
        mlflow.log_param("transformation_input", input_dataset)
        mlflow.log_param("transformation_output", output_dataset)
        mlflow.log_param("transformation_operations", ",".join(operations))
        
        # Store transformation proof
        proof_path = f"transform_proof_{output_dataset}.json"
        with open(proof_path, 'w') as f:
            json.dump(transform_proof.to_dict(), f, indent=2)
        mlflow.log_artifact(proof_path, "proofs")
        
        return transform_proof.id()
    
    def log_model_with_provenance(self, 
                                 model, 
                                 model_name: str,
                                 training_dataset: str,
                                 test_dataset: str = None):
        """Log model with complete dataset provenance"""
        
        # Get dataset history
        history = self.ledger.get_dataset_history(training_dataset)
        
        # Create provenance metadata
        provenance = {
            "training_dataset": training_dataset,
            "test_dataset": test_dataset,
            "dataset_lineage": [
                {
                    "dataset": event.dataset_name,
                    "operation": event.operation,
                    "timestamp": event.timestamp.isoformat(),
                    "proof_id": event.proof_id
                }
                for event in history
            ],
            "audit_trail_verified": self.ledger.verify_chain_integrity()
        }
        
        # Log model with provenance
        mlflow.sklearn.log_model(
            model,
            model_name,
            metadata={
                "zkp_provenance": provenance,
                "ledger_project": self.project_name
            }
        )
        
        # Generate and log compliance report
        audit_report = self.ledger.generate_audit_report(
            start_dataset=None,  # From beginning
            end_dataset=training_dataset,
            include_proofs=True
        )
        
        report_path = f"compliance_report_{model_name}.json"
        audit_report.export_json(report_path)
        mlflow.log_artifact(report_path, "compliance")


def main():
    print("🔬 MLflow + ZKP Dataset Ledger Integration Example")
    print("==================================================\n")
    
    # Set up MLflow
    mlflow.set_experiment("fraud-detection-zkp")
    
    # Initialize ZKP integration
    zkp_mlflow = MLflowZKPIntegration("fraud-detection-with-mlflow")
    
    with mlflow.start_run():
        print("📊 Starting MLflow run with ZKP integration...\n")
        
        # Step 1: Log raw dataset with proof
        print("1. Loading and proving raw dataset...")
        raw_data_path = "data/transactions.csv"
        
        # Create sample data if it doesn't exist
        create_sample_transactions_data(raw_data_path)
        
        raw_proof_id = zkp_mlflow.log_dataset_with_proof(
            raw_data_path,
            "raw-transactions",
            prove_properties={
                'row_count': True,
                'schema': True,
                'statistics': True,
                'private_columns': ['customer_id', 'account_number']  # Mark as private
            }
        )
        print(f"   ✅ Raw dataset proof ID: {raw_proof_id}")
        
        # Step 2: Data preprocessing with proof
        print("\n2. Preprocessing data with proof...")
        df = pd.read_csv(raw_data_path)
        
        # Simple preprocessing
        df_processed = df.copy()
        df_processed['amount_log'] = np.log1p(df['amount'])
        df_processed = pd.get_dummies(df_processed, columns=['merchant_category'])
        df_processed = df_processed.dropna()
        
        # Save processed data
        processed_path = "data/transactions_processed.csv"
        df_processed.to_csv(processed_path, index=False)
        
        # Log transformation with proof
        transform_proof_id = zkp_mlflow.log_transformation(
            "raw-transactions",
            "processed-transactions",
            ["log_transform", "one_hot_encode", "drop_nulls"],
            processed_path
        )
        print(f"   ✅ Transform proof ID: {transform_proof_id}")
        
        # Step 3: Train/test split with proof
        print("\n3. Creating train/test split with proof...")
        X = df_processed.drop(['fraud_label', 'transaction_id'], axis=1)
        y = df_processed['fraud_label']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Save splits (in practice, you'd use the ledger's split function)
        train_df = pd.concat([X_train, y_train], axis=1)
        test_df = pd.concat([X_test, y_test], axis=1)
        
        train_path = "data/train_set.csv"
        test_path = "data/test_set.csv"
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        
        # Log splits with proofs
        train_proof_id = zkp_mlflow.log_dataset_with_proof(
            train_path, "train-set", prove_properties={'row_count': True}
        )
        test_proof_id = zkp_mlflow.log_dataset_with_proof(
            test_path, "test-set", prove_properties={'row_count': True}
        )
        
        print(f"   ✅ Train set proof ID: {train_proof_id}")
        print(f"   ✅ Test set proof ID: {test_proof_id}")
        
        # Step 4: Train model
        print("\n4. Training model...")
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"   Model accuracy: {accuracy:.3f}")
        
        # Step 5: Log model with complete provenance
        print("\n5. Logging model with ZKP provenance...")
        zkp_mlflow.log_model_with_provenance(
            model,
            "fraud_detection_model",
            "train-set",
            "test-set"
        )
        
        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("n_estimators", 100)
        
        print("   ✅ Model logged with complete audit trail")
        
        # Step 6: Verify entire pipeline
        print("\n6. Verifying pipeline integrity...")
        chain_valid = zkp_mlflow.ledger.verify_chain_integrity()
        mlflow.log_metric("pipeline_verified", 1.0 if chain_valid else 0.0)
        
        print(f"   Pipeline integrity: {'✅ VERIFIED' if chain_valid else '❌ FAILED'}")
        
        print(f"\n🎉 MLflow run completed!")
        print(f"Run ID: {mlflow.active_run().info.run_id}")
        print(f"All datasets and transformations are cryptographically proven.")


def create_sample_transactions_data(file_path: str):
    """Create sample transaction data for the example"""
    import os
    os.makedirs("data", exist_ok=True)
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'transaction_id': range(1, n_samples + 1),
        'customer_id': np.random.randint(1000, 9999, n_samples),
        'account_number': np.random.randint(100000, 999999, n_samples),
        'amount': np.random.lognormal(3, 1, n_samples),
        'merchant_category': np.random.choice(['grocery', 'gas', 'restaurant', 'retail', 'online'], n_samples),
        'fraud_label': np.random.choice([0, 1], n_samples, p=[0.95, 0.05])  # 5% fraud
    }
    
    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False)
    print(f"   📝 Created sample data: {file_path}")


if __name__ == "__main__":
    main()