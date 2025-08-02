# ZKP Dataset Ledger Architecture

## Overview

The ZKP Dataset Ledger is designed as a modular system that provides cryptographic provenance and auditing capabilities for machine learning datasets. The architecture emphasizes privacy preservation, cryptographic security, and efficient proof generation/verification.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      ZKP Dataset Ledger                        │
├─────────────────────────────────────────────────────────────────┤
│                        CLI Interface                           │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐             │
│  │   Notarize  │ │  Transform  │ │    Audit    │             │
│  └─────────────┘ └─────────────┘ └─────────────┘             │
└─────────────────┬───────────────────────────────┬─────────────┘
                  │                               │
┌─────────────────▼─────────────────┐ ┌───────────▼─────────────┐
│         Core Library              │ │    Export Module        │
├───────────────────────────────────┤ ├─────────────────────────┤
│ ┌─────────────┐ ┌─────────────┐   │ │ ┌─────────┐ ┌─────────┐ │
│ │   Ledger    │ │   Crypto    │   │ │ │JSON-LD  │ │   PDF   │ │
│ │   Manager   │ │   Engine    │   │ │ │Reports  │ │Reports  │ │
│ └─────────────┘ └─────────────┘   │ │ └─────────┘ └─────────┘ │
│ ┌─────────────┐ ┌─────────────┐   │ └─────────────────────────┘
│ │   Dataset   │ │   Proof     │   │
│ │  Processor  │ │ Generator   │   │
│ └─────────────┘ └─────────────┘   │
└─────────────────┬─────────────────┘
                  │
┌─────────────────▼─────────────────┐
│        Storage Layer              │
├───────────────────────────────────┤
│ ┌─────────────┐ ┌─────────────┐   │
│ │   RocksDB   │ │ PostgreSQL  │   │
│ │  (Default)  │ │ (Optional)  │   │
│ └─────────────┘ └─────────────┘   │
└───────────────────────────────────┘
```

## Core Components

### 1. Ledger Manager

**Responsibility**: Maintains the append-only ledger of dataset operations using a Merkle tree structure.

**Key Features**:
- Immutable record keeping
- Cryptographic linking of operations
- Efficient proof generation for historical queries
- Branch and merge support for complex workflows

**Implementation**: `src/ledger.rs`

### 2. Crypto Engine

**Responsibility**: Handles all cryptographic operations including zero-knowledge proof generation and verification.

**Key Features**:
- Groth16 proof system implementation
- BLS12-381 elliptic curve operations
- Merkle tree hash computations
- Privacy-preserving statistical proofs

**Implementation**: `src/crypto/` module

**Sub-components**:
- `hash.rs`: SHA-3 and Blake3 hashing
- `merkle.rs`: Merkle tree operations
- `circuits.rs`: ZK circuit definitions

### 3. Dataset Processor

**Responsibility**: Handles dataset ingestion, validation, and transformation tracking.

**Key Features**:
- Multiple format support (CSV, Parquet, JSON)
- Streaming processing for large datasets
- Schema validation and inference
- Transformation operation recording

**Implementation**: `src/dataset.rs`

### 4. Proof Generator

**Responsibility**: Creates zero-knowledge proofs for dataset properties without revealing sensitive data.

**Key Features**:
- Row count proofs
- Schema compliance proofs
- Statistical property proofs
- Transformation correctness proofs

**Implementation**: `src/proof.rs`

### 5. Storage Layer

**Responsibility**: Provides persistent storage with pluggable backend support.

**Supported Backends**:
- **RocksDB** (default): High-performance key-value store
- **PostgreSQL** (optional): Relational database with JSON support

**Implementation**: `src/storage.rs`

## Data Flow

### 1. Dataset Notarization Flow

```
Dataset File → Hash Computation → Merkle Tree → ZK Circuit → Groth16 Proof → Ledger Entry
     ↓               ↓              ↓           ↓             ↓              ↓
   Validate       SHA-3/Blake3    Build Tree   Generate      Create         Store to
   Schema         Content Hash    Structure    Constraints   Proof          Backend
```

### 2. Verification Flow

```
Proof + Public Inputs → Verification Engine → Cryptographic Validation → Result
       ↓                       ↓                      ↓                   ↓
   Load from Storage      Groth16 Verify         Check Constraints    Valid/Invalid
```

### 3. Audit Trail Query Flow

```
Query Parameters → Ledger Query → Merkle Proof → Verification → Audit Report
       ↓               ↓             ↓             ↓              ↓
   Time Range/       Search         Generate      Validate       JSON-LD/PDF
   Dataset ID        Ledger         Chain Proof   Chain          Export
```

## Security Architecture

### Cryptographic Guarantees

1. **Integrity**: SHA-3 hashing ensures tamper detection
2. **Immutability**: Merkle tree structure prevents historical modification
3. **Privacy**: Zero-knowledge proofs hide sensitive data
4. **Authenticity**: Digital signatures on ledger entries

### Threat Model

**Protected Against**:
- Data tampering and unauthorized modifications
- Privacy breaches during auditing
- False claims about dataset properties
- Replay attacks on proof submissions

**Assumptions**:
- Trusted setup for Groth16 is conducted securely
- Private keys are kept secure
- System clock is reasonably accurate
- Storage backend is not compromised

### Security Controls

1. **Input Validation**: All inputs sanitized and validated
2. **Memory Safety**: Rust's ownership system prevents memory vulnerabilities  
3. **Constant-Time Operations**: Cryptographic operations resist timing attacks
4. **Secure Random Generation**: Using system entropy for key generation

## Performance Characteristics

### Scalability Targets

| Operation | Dataset Size | Target Time | Memory Usage |
|-----------|-------------|-------------|--------------|
| Notarize | 1M rows | <5 seconds | <1GB |
| Transform | 10M rows | <30 seconds | <2GB |
| Verify | Any size | <100ms | <10MB |
| Audit Query | 1B operations | <1 second | <100MB |

### Optimization Strategies

1. **Streaming Processing**: Handle datasets larger than available memory
2. **Parallel Proof Generation**: Utilize multiple CPU cores
3. **Proof Caching**: Reuse intermediate computations
4. **Batch Operations**: Group multiple dataset operations

## API Design

### CLI Interface

```bash
zkp-ledger <command> [options]

Commands:
  init        Initialize new ledger
  notarize    Record dataset with proof
  transform   Record transformation operation  
  split       Record dataset split operation
  verify      Verify existing proof
  audit       Generate audit report
  export      Export proofs/reports
```

### Rust Library API

```rust
pub struct Ledger {
    storage: Box<dyn Storage>,
    crypto: CryptoEngine,
}

impl Ledger {
    pub fn new(config: LedgerConfig) -> Result<Self>;
    pub fn notarize_dataset(&mut self, dataset: Dataset, name: &str) -> Result<Proof>;
    pub fn verify_proof(&self, proof: &Proof) -> Result<bool>;
    pub fn get_audit_trail(&self, query: AuditQuery) -> Result<Vec<LedgerEntry>>;
}
```

### Python Bindings API

```python
class Ledger:
    def __init__(self, project_name: str)
    def notarize_dataset(self, path: str, name: str, **kwargs) -> DatasetProof
    def verify_proof(self, proof: DatasetProof) -> bool  
    def generate_audit_report(self, **kwargs) -> AuditReport
```

## Configuration Management

### Ledger Configuration

```toml
[ledger]
name = "production-ml-pipeline"
hash_algorithm = "sha3-256"  # or "blake3"
proof_system = "groth16"
compression = true

[storage]
backend = "rocksdb"  # or "postgres"
path = "./ledger-data"
max_size_gb = 100

[proof]
curve = "bls12-381"
security_level = 128
parallel_prove = true
cache_size_mb = 1024

[export]
formats = ["json-ld", "pdf", "html"]
include_visualizations = true
```

## Deployment Architecture

### Standalone Deployment

```
┌─────────────────┐
│   Application   │
├─────────────────┤
│ ZKP Ledger CLI  │
├─────────────────┤
│   RocksDB       │
│   Local Storage │
└─────────────────┘
```

### Distributed Deployment

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Client App    │    │   Web Service   │    │   Database      │
├─────────────────┤    ├─────────────────┤    ├─────────────────┤
│ ZKP Ledger API  │───▶│ ZKP Ledger Core │───▶│   PostgreSQL    │
│ (Python/Rust)   │    │ (HTTP/gRPC)     │    │   Cluster       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Integration Patterns

### MLflow Integration

```python
# Automatic dataset tracking with proofs
with mlflow.start_run():
    proof = ledger.notarize_dataset("train.csv", "training-v1")
    mlflow.log_artifact(proof.export_json(), "dataset-proof.json")
    
    model = train_model(data)
    mlflow.sklearn.log_model(model, "model", 
        metadata={"dataset_proof": proof.hash})
```

### CI/CD Pipeline Integration

```yaml
- name: Validate Dataset Provenance
  run: |
    zkp-ledger verify-chain --from ${{ env.DATASET_VERSION }}
    zkp-ledger audit --output compliance-report.json
```

## Future Architecture Considerations

### Planned Enhancements

1. **Federated Learning Support**: Multi-party computation capabilities
2. **Differential Privacy**: Integrated noise mechanisms  
3. **Smart Contract Integration**: Blockchain-based verification
4. **Advanced Circuits**: Support for more complex statistical proofs
5. **Multi-Language SDKs**: JavaScript, Go, and Java bindings

### Scalability Roadmap

1. **Phase 1**: Single-node optimization (current)
2. **Phase 2**: Distributed storage and computation
3. **Phase 3**: Decentralized network deployment
4. **Phase 4**: Cross-chain interoperability

## Development Guidelines

### Code Organization

```
src/
├── lib.rs              # Public API definitions
├── ledger.rs           # Core ledger implementation
├── dataset.rs          # Dataset processing
├── proof.rs            # Proof generation/verification
├── storage.rs          # Storage abstraction
├── error.rs            # Error handling
├── crypto/             # Cryptographic primitives
│   ├── mod.rs
│   ├── hash.rs
│   ├── merkle.rs
│   └── circuits.rs
└── bin/                # CLI implementation
    └── cli.rs
```

### Testing Strategy

1. **Unit Tests**: Each module thoroughly tested
2. **Integration Tests**: End-to-end workflows
3. **Property Tests**: Cryptographic invariants
4. **Benchmarks**: Performance regression detection
5. **Security Tests**: Adversarial scenarios

### Dependencies

**Core Dependencies**:
- `arkworks-rs`: Zero-knowledge proof system
- `serde`: Serialization framework
- `polars`: High-performance data processing
- `rocksdb`: Default storage backend
- `tokio`: Async runtime

**Security Dependencies**:
- `sha3`: Cryptographic hashing
- `blake3`: Fast cryptographic hashing
- `ring`: Additional crypto primitives
- `zeroize`: Secure memory clearing

This architecture provides a solid foundation for cryptographic dataset auditing while maintaining flexibility for future enhancements and integrations.