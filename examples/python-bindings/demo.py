#!/usr/bin/env python3
"""
Python Bindings Demo for ZKP Dataset Ledger

This example shows how to use the Python bindings to interact
with the ZKP Dataset Ledger from Python applications.

Demonstrates:
- Basic ledger operations
- Privacy-preserving proofs
- Statistical property verification
- Audit trail generation
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path

# Import Python bindings (installed via pip install zkp-dataset-ledger)
try:
    from zkp_dataset_ledger import (
        Ledger, 
        DatasetProof, 
        ProofConfig,
        StatisticalProperties,
        PrivacyLevel
    )
except ImportError:
    print("⚠️  Python bindings not available.")
    print("Install with: pip install zkp-dataset-ledger")
    print("Or build with: maturin develop --release")
    exit(1)


def main():
    print("🐍 ZKP Dataset Ledger - Python Bindings Demo")
    print("=============================================\n")

    # Initialize ledger
    ledger = Ledger("python-demo-project")
    print("✅ Initialized ledger for Python demo\n")

    # Create sample dataset
    sample_data = create_sample_dataset()
    data_path = "demo_data.csv"
    sample_data.to_csv(data_path, index=False)
    print(f"📊 Created sample dataset: {data_path}")
    print(f"   Shape: {sample_data.shape}")
    print(f"   Columns: {list(sample_data.columns)}\n")

    # Basic proof generation
    print("1. Basic Dataset Proof")
    print("----------------------")
    
    basic_config = ProofConfig(
        prove_row_count=True,
        prove_column_count=True,
        prove_schema=True
    )
    
    basic_proof = ledger.notarize_dataset_file(data_path, "demo-dataset-v1", basic_config)
    
    print(f"   Proof ID: {basic_proof.id()}")
    print(f"   Dataset hash: {basic_proof.dataset_hash()}")
    print(f"   Proof size: {basic_proof.size_bytes()} bytes")
    print(f"   Is valid: {ledger.verify_proof(basic_proof)}\n")

    # Privacy-preserving statistical proof
    print("2. Privacy-Preserving Statistical Proof")
    print("----------------------------------------")
    
    privacy_config = ProofConfig(
        prove_row_count=True,
        prove_statistical_properties=True,
        privacy_level=PrivacyLevel.HIGH,
        private_columns=["salary", "ssn"]  # Mark sensitive columns
    )
    
    stats_proof = ledger.notarize_dataset_file(
        data_path, 
        "demo-dataset-private", 
        privacy_config
    )
    
    print(f"   Private proof ID: {stats_proof.id()}")
    print(f"   Privacy level: HIGH")
    print(f"   Protected columns: salary, ssn")
    print(f"   Statistical properties proven without revealing data")
    print(f"   Proof size: {stats_proof.size_bytes()} bytes\n")

    # Prove specific statistical properties
    print("3. Custom Statistical Property Proofs")
    print("-------------------------------------")
    
    # Prove mean age without revealing individual ages
    age_stats_proof = ledger.prove_column_statistics(
        "demo-dataset-v1",
        "age",
        StatisticalProperties(
            prove_mean=True,
            prove_variance=True,
            prove_min_max_range=True,
            differential_privacy_epsilon=1.0  # Add DP noise
        )
    )
    
    print(f"   Age statistics proof: {age_stats_proof.id()}")
    print(f"   Proven: mean, variance, range (with differential privacy)")
    
    # Prove data quality metrics
    quality_proof = ledger.prove_data_quality(
        "demo-dataset-v1",
        {
            "null_count": True,
            "duplicate_count": True,
            "outlier_count": True,
            "data_types_valid": True
        }
    )
    
    print(f"   Data quality proof: {quality_proof.id()}")
    print(f"   Proven: nulls, duplicates, outliers, type validity\n")

    # Data transformation with proof
    print("4. Data Transformation with Proof")
    print("----------------------------------")
    
    # Simulate data cleaning
    cleaned_data = sample_data.copy()
    cleaned_data = cleaned_data.dropna()
    cleaned_data["age_normalized"] = (cleaned_data["age"] - cleaned_data["age"].mean()) / cleaned_data["age"].std()
    cleaned_data["salary_log"] = np.log1p(cleaned_data["salary"])
    
    cleaned_path = "demo_data_cleaned.csv"
    cleaned_data.to_csv(cleaned_path, index=False)
    
    # Record transformation with proof
    transform_proof = ledger.record_transformation_file(
        "demo-dataset-v1",
        "demo-dataset-cleaned",
        ["drop_nulls", "normalize_age", "log_transform_salary"],
        cleaned_path
    )
    
    print(f"   Transform proof: {transform_proof.id()}")
    print(f"   Operations: drop_nulls, normalize_age, log_transform_salary")
    print(f"   Input rows: {len(sample_data)}")
    print(f"   Output rows: {len(cleaned_data)}\n")

    # Federated learning scenario
    print("5. Federated Learning Proof")
    print("----------------------------")
    
    # Simulate multiple parties with data
    party_datasets = create_federated_datasets()
    
    fed_proofs = []
    for i, (party_name, party_data) in enumerate(party_datasets.items()):
        party_path = f"party_{i+1}_data.csv"
        party_data.to_csv(party_path, index=False)
        
        # Each party proves their data properties locally
        party_proof = ledger.notarize_dataset_file(
            party_path,
            f"party-{i+1}-data",
            ProofConfig(
                prove_row_count=True,
                prove_schema=True,
                privacy_level=PrivacyLevel.HIGH
            )
        )
        fed_proofs.append(party_proof)
        print(f"   Party {i+1} ({party_name}): {party_proof.id()}")
    
    # Aggregate proofs without sharing raw data
    aggregated_proof = ledger.aggregate_federated_proofs(
        fed_proofs,
        aggregation_type="secure_sum"
    )
    
    print(f"   Aggregated proof: {aggregated_proof.id()}")
    print(f"   Total participants: {len(fed_proofs)}")
    print(f"   Data remains private to each party\n")

    # Generate comprehensive audit report
    print("6. Comprehensive Audit Report")
    print("------------------------------")
    
    audit_report = ledger.generate_audit_report(
        start_dataset=None,  # From beginning
        end_dataset=None,    # To latest
        include_proofs=True,
        include_verification=True
    )
    
    # Export in different formats
    audit_report.export_json("python_demo_audit.json")
    audit_report.export_html("python_demo_audit.html")
    
    print("   Generated audit reports:")
    print("   - python_demo_audit.json")
    print("   - python_demo_audit.html")
    
    # Print summary statistics
    summary = audit_report.get_summary()
    print(f"\n   Audit Summary:")
    print(f"   - Total datasets: {summary['total_datasets']}")
    print(f"   - Total transformations: {summary['total_transformations']}")
    print(f"   - Total proofs: {summary['total_proofs']}")
    print(f"   - Chain integrity: {'✅ Valid' if summary['chain_valid'] else '❌ Invalid'}")

    # Demonstrate verification by external party
    print("\n7. External Verification")
    print("-------------------------")
    
    all_proofs = [basic_proof, stats_proof, age_stats_proof, quality_proof, transform_proof, aggregated_proof]
    
    print("   Verifying all proofs independently:")
    for i, proof in enumerate(all_proofs, 1):
        is_valid = ledger.verify_proof(proof)
        proof_type = type(proof).__name__
        print(f"   {i}. {proof_type}: {'✅ Valid' if is_valid else '❌ Invalid'}")
    
    # Verify entire chain
    chain_valid = ledger.verify_chain_integrity()
    print(f"\n   Complete chain verification: {'✅ Valid' if chain_valid else '❌ Invalid'}")

    print("\n🎉 Python bindings demo completed successfully!")
    print("All operations completed with cryptographic proofs.")


def create_sample_dataset():
    """Create a sample dataset for demonstration"""
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'employee_id': range(1, n_samples + 1),
        'age': np.random.normal(35, 10, n_samples).astype(int),
        'salary': np.random.lognormal(10.5, 0.5, n_samples).astype(int),
        'department': np.random.choice(['Engineering', 'Sales', 'Marketing', 'HR', 'Finance'], n_samples),
        'years_experience': np.random.exponential(5, n_samples).astype(int),
        'ssn': [f"{np.random.randint(100,999)}-{np.random.randint(10,99)}-{np.random.randint(1000,9999)}" 
                for _ in range(n_samples)],
        'performance_score': np.random.normal(3.5, 0.8, n_samples)
    }
    
    # Add some nulls and outliers for realistic data
    df = pd.DataFrame(data)
    df.loc[np.random.choice(df.index, 50, replace=False), 'performance_score'] = np.nan
    df.loc[np.random.choice(df.index, 20, replace=False), 'salary'] *= 3  # Outliers
    
    return df


def create_federated_datasets():
    """Create sample datasets for federated learning scenario"""
    np.random.seed(123)
    
    datasets = {}
    
    # Hospital A data
    datasets['hospital_a'] = pd.DataFrame({
        'patient_id': range(1, 501),
        'age': np.random.normal(45, 15, 500).astype(int),
        'diagnosis': np.random.choice(['diabetes', 'hypertension', 'normal'], 500),
        'treatment_outcome': np.random.choice(['improved', 'stable', 'declined'], 500)
    })
    
    # Hospital B data  
    datasets['hospital_b'] = pd.DataFrame({
        'patient_id': range(501, 851),
        'age': np.random.normal(50, 12, 350).astype(int),
        'diagnosis': np.random.choice(['diabetes', 'hypertension', 'normal'], 350),
        'treatment_outcome': np.random.choice(['improved', 'stable', 'declined'], 350)
    })
    
    # Research Center data
    datasets['research_center'] = pd.DataFrame({
        'patient_id': range(851, 1001),
        'age': np.random.normal(40, 20, 150).astype(int),
        'diagnosis': np.random.choice(['diabetes', 'hypertension', 'normal'], 150),
        'treatment_outcome': np.random.choice(['improved', 'stable', 'declined'], 150)
    })
    
    return datasets


if __name__ == "__main__":
    main()