# ADR-001: Use Groth16 for Zero-Knowledge Proofs

## Status

**Accepted**

**Date**: 2024-01-15
**Author**: ZKP Dataset Ledger Team
**Reviewers**: Cryptography Review Committee

## Context

The ZKP Dataset Ledger requires a zero-knowledge proof system to enable privacy-preserving verification of dataset properties without revealing the underlying data. The choice of proof system is critical as it affects performance, security, proof size, and developer experience.

### Background
- Need to prove statistical properties of datasets without data disclosure
- Support for custom circuits defining dataset validation rules
- Integration with Rust ecosystem for type safety and performance
- Compatibility with existing cryptographic libraries

### Problem Statement
We need to select a zero-knowledge proof system that balances:
- Proof generation performance for large datasets
- Verification speed for real-time use cases
- Proof size for network transmission and storage
- Security guarantees and cryptographic maturity
- Developer ecosystem and library availability

## Decision Drivers

- **Performance**: Sub-minute proof generation for 1M row datasets
- **Verification Speed**: <100ms verification regardless of dataset size
- **Proof Size**: <1KB for efficient storage and transmission
- **Security**: Well-analyzed cryptographic assumptions
- **Ecosystem**: Mature Rust implementations available
- **Flexibility**: Support for custom circuit development

## Considered Options

### Option 1: Groth16
**Description**: Succinct non-interactive argument of knowledge with trusted setup

**Pros**:
- Constant-size proofs (~288 bytes)
- Fast verification (~10ms)
- Mature cryptographic analysis
- Excellent Rust ecosystem (arkworks)
- Wide industry adoption

**Cons**:
- Requires trusted setup ceremony
- Setup specific to each circuit
- Not quantum-resistant
- Limited post-quantum migration path

**Trade-offs**:
- Accepts trusted setup burden for superior performance

### Option 2: PLONK/PLONKish
**Description**: Universal SNARK with single trusted setup

**Pros**:
- Universal trusted setup (circuit-agnostic)
- Good proof sizes (~1KB)
- Active development and research
- Growing Rust ecosystem

**Cons**:
- Larger proofs than Groth16
- Less mature than Groth16
- More complex implementation
- Higher verification costs

**Trade-offs**:
- Universal setup convenience vs. proof size/speed

### Option 3: STARKs (Transparent)
**Description**: Transparent (no trusted setup) succinct arguments

**Pros**:
- No trusted setup required
- Post-quantum security
- Transparent and verifiable
- Excellent for large computations

**Cons**:
- Large proof sizes (100KB+)
- Slower verification
- Limited Rust ecosystem
- Complex circuit development

**Trade-offs**:
- Transparency vs. practical efficiency

### Option 4: Bulletproofs
**Description**: Short non-interactive zero-knowledge proofs

**Pros**:
- No trusted setup
- Logarithmic proof size
- Good for range proofs
- Mature implementations

**Cons**:
- Linear verification time
- Limited to specific proof types
- Poor scaling for complex circuits
- Slower than SNARKs

**Trade-offs**:
- Setup-free vs. verification performance

## Decision

We decided to implement **Groth16** because:

1. **Performance Requirements**: Our target of <30s proof generation and <100ms verification for production workloads strongly favors Groth16's constant-time verification and optimized implementations.

2. **Proof Size Constraints**: At ~288 bytes, Groth16 proofs enable efficient storage in the Merkle tree ledger and fast network transmission for audit reports.

3. **Ecosystem Maturity**: The arkworks-rs ecosystem provides production-ready, well-audited implementations with extensive optimization and hardware acceleration support.

4. **Security Track Record**: Groth16 has undergone extensive cryptographic analysis and is deployed in production systems like Zcash, providing confidence in its security properties.

### Implementation Details
- Use `ark-groth16` as the core proving system
- Leverage `ark-bls12-381` for the elliptic curve implementation
- Support custom R1CS circuits via `ark-relations`
- Implement trusted setup parameter management and verification

## Consequences

### Positive Consequences
- **Excellent Performance**: Constant-time verification enables real-time audit verification
- **Minimal Storage Overhead**: Tiny proofs reduce ledger storage requirements
- **Rich Ecosystem**: Access to optimized implementations, hardware acceleration, and extensive tooling
- **Production Readiness**: Battle-tested cryptography with known security properties

### Negative Consequences
- **Trusted Setup Dependency**: Requires ceremony for each circuit type, introducing operational complexity
- **Post-Quantum Vulnerability**: Future quantum computers could break the cryptographic assumptions
- **Circuit Specificity**: Each new proof type requires its own trusted setup
- **Setup Verification Burden**: Users must verify trusted setup integrity

### Neutral Consequences
- **Migration Path**: Future transition to universal SNARKs or post-quantum systems will require significant refactoring
- **Developer Learning Curve**: Team needs expertise in R1CS circuit development and trusted setup procedures

## Implementation

### Action Items
- [x] Task 1: Integrate arkworks-rs dependencies (Owner: Core Team)
- [x] Task 2: Implement basic proof generation pipeline (Owner: Crypto Team)
- [x] Task 3: Create circuit library for common dataset properties (Owner: Crypto Team)
- [ ] Task 4: Establish trusted setup ceremony procedures (Owner: Security Team)
- [ ] Task 5: Implement setup parameter verification (Owner: Security Team)

### Timeline
- **Phase 1** (Week 1-2): Core integration and basic proof types
- **Phase 2** (Week 3-4): Advanced circuits and performance optimization
- **Phase 3** (Week 5-6): Trusted setup tooling and security hardening

### Success Criteria
- **Performance**: Proof generation <30s for 1M row datasets
- **Verification**: <100ms verification time consistently
- **Proof Size**: <1KB for all supported proof types
- **Security**: Clean security audit of implementation

## Related Decisions

- **Blocks**: [ADR-005](005-merkle-tree-ledger-structure.md) - Proof storage in ledger structure
- **Related to**: [ADR-002](002-rocksdb-default-storage.md) - Storage of trusted setup parameters

## References

- [Groth16 Paper]: "On the Size of Pairing-based Non-interactive Arguments" by Jens Groth
- [Arkworks Documentation]: https://arkworks.rs/
- [Zcash Protocol Specification]: https://zips.z.cash/protocol/protocol.pdf
- [Trusted Setup Ceremonies]: https://github.com/ethereum/kzg-ceremony

## Revision History

| Date | Author | Change |
|------|--------|--------|
| 2024-01-15 | ZKP Team | Initial version |
| 2024-01-20 | Security Team | Added trusted setup considerations |