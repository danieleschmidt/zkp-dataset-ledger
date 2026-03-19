# zkp-dataset-ledger

> **Zero-knowledge proof ledger for ML dataset provenance**
>
> Prove your training data has the right properties — without revealing the data.

[![Rust](https://img.shields.io/badge/rust-1.70%2B-orange)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-33%20passing-green)](#)

---

## The Problem

When training ML models, **data provenance matters**:

- Regulators want to verify GDPR compliance without accessing personal data
- Auditors want to confirm a model wasn't trained on prohibited datasets
- Model consumers want assurance about training data quality
- Federated learning parties want to prove their contribution without sharing it

The naive solution — share the dataset — violates privacy, IP protection, and data governance.

## The Solution: Zero-Knowledge Proofs

A **zero-knowledge proof (ZKP)** lets you prove a statement is true *without revealing why* it's true.

This library applies ZKP primitives to dataset provenance:

```
"My training dataset has exactly 1 million rows, 
 mean age < 40, and was collected after 2020"
       ↓
[ZKP proves this to an auditor]
       ↓
The auditor learns ONLY those facts.
The actual dataset stays private.
```

---

## Cryptographic Primitives

### 1. Hash Commitment Scheme (Pedersen-style)

A **commitment** binds the prover to a dataset without revealing it:

```
C = SHA256("commit" || SHA256(dataset) || nonce)
```

- **Binding**: The prover cannot change the dataset after publishing `C`  
  (would require a SHA-256 collision — computationally infeasible)
- **Hiding**: Given `C` alone, the verifier learns nothing about the dataset  
  (the 256-bit random `nonce` acts as a blinding factor)

### 2. Non-Interactive Sigma Protocol (Fiat-Shamir Transform)

A **Sigma protocol** is a 3-round interactive proof that the prover knows a secret. The **Fiat-Shamir transform** makes it non-interactive by replacing the verifier's random challenge with a deterministic hash of the transcript:

```
challenge = SHA256("proof-v1" || commitment || proof_type || witness)
response  = SHA256("response-v1" || nonce || challenge || dataset_hash)
```

The verifier checks:
1. The challenge matches the expected transcript hash (no tampering)
2. The claimed property is consistent with the witness
3. All values are well-formed

**Soundness**: A cheating prover cannot forge a valid `response` without knowing the `nonce`, since that would require inverting SHA-256.

### 3. Tamper-Evident Ledger (Hash Chaining)

The ledger links entries with a chain hash (similar to blockchain headers):

```
Entry 0               Entry 1               Entry 2
┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│ commitment₀  │      │ commitment₁  │      │ commitment₂  │
│ proofs       │      │ proofs       │      │ proofs       │
│ prev="genesis│ ───> │ prev=H(e₀)  │ ───> │ prev=H(e₁)  │
└──────────────┘      └──────────────┘      └──────────────┘
```

Modifying any entry invalidates all subsequent chain hashes.

---

## Provable Properties

| Property | Description | Example |
|----------|-------------|---------|
| `Existence` | Prove the dataset exists and matches the commitment | "I have dataset D" |
| `ExactRowCount` | Prove exactly N rows | "My dataset has 10,000 rows" |
| `MinRowCount` | Prove at least N rows | "At least 1,000 training examples" |
| `MaxRowCount` | Prove at most N rows | "No more than 100,000 rows" |
| `RowCountRange` | Prove row count is in [min, max] | "Between 500 and 5,000 rows" |
| `ColumnCount` | Prove the number of columns | "5-dimensional feature space" |
| `SchemaFingerprint` | Prove column names match a known schema | "Same schema as reference dataset" |
| `StatisticalBounds` | Prove mean of a column is in [a, b] | "Mean age between 25 and 65" |
| `CommittedBefore` | Prove the dataset was committed before a timestamp | "Data collected before GDPR cutoff" |

---

## Quick Start

```bash
git clone https://github.com/danieleschmidt/zkp-dataset-ledger
cd zkp-dataset-ledger
cargo build --release
```

### CLI Usage

```bash
# 1. Commit to your dataset (keep the output private — it contains the nonce)
cargo run -- commit my_training_data.csv --description "MNIST subset" --output commitment.json

# 2. Generate a ZK proof of a property (safe to share)
cargo run -- prove commitment.json my_training_data.csv \
  --property row-count=60000 \
  --output proof.json

# 3. Auditor verifies the proof (no dataset needed)
cargo run -- verify proof.json

# 4. Run the interactive demo
cargo run -- demo
```

### Library Usage

```rust
use zkp_dataset_ledger::{
    commitment::{CommitmentConfig, DatasetCommitment},
    proof::{ProofGenerator, ProofType, ProofVerifier},
    ledger::DatasetLedger,
};

// Load your dataset
let dataset = std::fs::read("training_data.csv")?;

// Step 1: Data owner commits
let commitment = DatasetCommitment::commit(
    &dataset,
    &CommitmentConfig::default(),
    Some("ImageNet subset v2".to_string()),
)?;

// Publish commitment_hash to an audit log
println!("Public commitment: {}", commitment.commitment_hash);

// Step 2: Generate ZK proofs
let proof = ProofGenerator::generate(
    &commitment,
    &dataset,
    ProofType::StatisticalBounds {
        column: "label_count".to_string(),
        mean_min: 950.0,
        mean_max: 1050.0,
    },
)?;

// Step 3: Auditor verifies (no dataset required)
assert!(ProofVerifier::verify(&proof)?);

// Step 4: Append to tamper-evident ledger
let mut ledger = DatasetLedger::new("model-audit-trail");
ledger.append_with_proofs(commitment, vec![proof])?;
assert!(ledger.verify_integrity());
```

---

## Architecture

```
src/
├── lib.rs          — Public API and module declarations
├── crypto/
│   └── mod.rs      — SHA-256 primitives, commitment scheme, Fiat-Shamir
├── commitment.rs   — DatasetCommitment: commit, verify, selective disclosure
├── proof.rs        — ProofGenerator, ProofVerifier, ProofType enum
└── ledger.rs       — DatasetLedger: append-only log with chain integrity
```

---

## Security Notes

This implementation uses **standard SHA-256** rather than elliptic curve arithmetic. It demonstrates the ZKP protocol structure clearly using familiar primitives.

**For production use**, consider:

- **Groth16** or **PLONK** circuits with BLS12-381 or BN254 curves — stronger soundness and more succinct proofs
- **bulletproofs** — efficient range proofs without trusted setup
- **RISC Zero** or **SP1** — general-purpose zkVM for complex statement proofs
- The [`arkworks`](https://github.com/arkworks-rs) or [`bellman`](https://github.com/zkcrypto/bellman) Rust crates

This library is intended as an educational foundation and can be extended with those primitives.

---

## Relation to ML Governance

This work connects to the broader AI governance agenda:

| Challenge | ZKP Solution |
|-----------|-------------|
| "Prove this model wasn't trained on my data" | Dataset existence commitment: prove hash does/doesn't match |
| "Prove training data meets GDPR age restrictions" | Statistical bounds proof on age column |
| "Prove federated learning contribution" | Each party commits to their shard independently |
| "Audit ML supply chain" | Tamper-evident ledger links dataset → model → deployment |
| "Prove bias metrics without dataset disclosure" | Statistical bounds on protected attribute distributions |

**Companion reading**: "Zero-Knowledge Provenance for Knowledge Graph Construction" explores applying these primitives to knowledge graph ML pipelines.

---

## Running Tests

```bash
cargo test
```

**33 tests** across:
- Cryptographic primitives (commitment scheme, Fiat-Shamir)
- Proof generation and verification (all 9 proof types)
- Tamper-detection (modified proofs, modified ledgers)
- Integration workflow (end-to-end commit → prove → verify → ledger)
- Edge cases (dishonest prover, out-of-range claims)

---

## License

Apache-2.0 — see [LICENSE](LICENSE)
