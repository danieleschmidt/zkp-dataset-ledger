# ZKP Dataset Ledger Examples

This directory contains practical examples demonstrating real-world usage of the ZKP Dataset Ledger.

## 📁 Example Categories

### 🚀 Basic Usage (`basic/`)
**Purpose**: Learn fundamental concepts and operations
- Dataset notarization with zero-knowledge proofs
- Recording data transformations
- Creating auditable train/test splits
- Proof verification and chain integrity
- Generating compliance reports

**Run**: `cargo run --example basic`

### 🧪 MLflow Integration (`mlflow/`)
**Purpose**: Production ML experiment tracking with cryptographic provenance
- Automatic dataset proof generation in MLflow runs
- Model metadata with complete dataset lineage
- Compliance report generation for auditors
- Proof verification during model deployment

**Setup**:
```bash
cd examples/mlflow
pip install -r requirements.txt
python integration.py
```

### 🐍 Python Bindings (`python-bindings/`)
**Purpose**: Using ZKP Dataset Ledger from Python applications
- Privacy-preserving statistical proofs
- Custom property verification 
- Federated learning scenarios
- Data quality attestation
- Multi-format audit reports

**Run**: `python examples/python-bindings/demo.py`

### ☸️ Kubeflow Integration (`kubeflow/`)
**Purpose**: Kubernetes-native ML pipelines with ZKP auditing
- Pipeline step verification
- Distributed proof generation
- Container-based dataset processing
- Kubernetes secret management

**Deploy**: `kubectl apply -f examples/kubeflow/`

## 🎯 Use Case Examples

### Regulatory Compliance
```bash
# Generate EU AI Act compliance report
zkp-ledger audit --standard eu-ai-act --output compliance-report.pdf

# GDPR-compliant data processing proof
zkp-ledger notarize --privacy-level high --gdpr-compliant dataset.csv
```

### Federated Learning
```python
# Prove dataset properties across institutions without sharing data
fed_ledger = FederatedLedger(participants=["hospital-a", "hospital-b"])
global_proof = fed_ledger.aggregate_proofs()
```

### Model Cards & Documentation
```bash
# Generate model card with cryptographic dataset provenance
zkp-ledger generate-model-card --dataset train-v1 --format json-ld
```

## 🔧 Development Examples

### Custom ZK Circuits
```rust
// examples/custom-circuits/fairness.rs
impl Circuit for FairnessCircuit {
    fn generate_constraints(&self, cs: &mut ConstraintSystem) -> Result<()> {
        // Prove demographic parity without revealing sensitive attributes
        let group_stats = self.compute_group_statistics();
        cs.enforce_fairness_constraint(group_stats)?;
        Ok(())
    }
}
```

### Storage Backend Integration
```rust
// examples/storage/s3-backend.rs
let ledger = Ledger::builder()
    .storage(S3Storage::new("my-bucket", "ledger/"))
    .build()?;
```

## 📊 Performance Benchmarks

Run comprehensive benchmarks across different dataset sizes:

```bash
cd examples/benchmarks
cargo bench --features benchmarks

# Generate performance report
cargo run --bin benchmark-report
```

## 🏗️ Integration Templates

### CI/CD Pipeline Integration
```yaml
# .github/workflows/ml-pipeline.yml
- name: Prove dataset integrity
  run: zkp-ledger verify --chain --strict
  
- name: Generate audit artifacts
  run: zkp-ledger audit --format json-ld --output ${{ github.sha }}-audit.json
```

### Docker Integration
```dockerfile
# Dockerfile.ml-pipeline
FROM zkpdatasetledger/zkp-ledger:latest
COPY datasets/ /data/
RUN zkp-ledger notarize /data/train.csv --name production-training-v1
```

## 🤝 Contributing Examples

We welcome new examples! Priority areas:

- **Framework Integrations**: TensorFlow, PyTorch, Hugging Face
- **Cloud Platforms**: AWS SageMaker, Google AI Platform, Azure ML
- **Data Formats**: Parquet, HDF5, Arrow, Delta Lake
- **Compliance Standards**: SOX, HIPAA, ISO 27001
- **Edge Computing**: TensorFlow Lite, ONNX Runtime

### Adding New Examples

1. Create directory: `examples/your-integration/`
2. Include README with clear setup instructions
3. Add practical, real-world scenario
4. Test with CI/CD pipeline
5. Document performance characteristics

### Example Structure
```
examples/your-integration/
├── README.md           # Setup and usage instructions
├── main.rs/.py         # Primary example code
├── Cargo.toml          # Dependencies (if Rust)
├── requirements.txt    # Dependencies (if Python)
├── sample-data/        # Test datasets
└── expected-output/    # Expected results for testing
```

## 📚 Learning Path

**Beginner**: Start with `basic/` → `python-bindings/`
**Intermediate**: Try `mlflow/` → `kubeflow/`
**Advanced**: Implement custom circuits and storage backends

## 🔗 Related Resources

- [Main Documentation](../README.md)
- [API Reference](https://docs.rs/zkp-dataset-ledger)
- [Performance Benchmarks](../benches/)
- [Contributing Guide](../CONTRIBUTING.md)
- [Security Policy](../SECURITY.md)