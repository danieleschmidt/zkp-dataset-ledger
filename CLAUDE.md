# Claude Agent Configuration for ZKP Dataset Ledger

## Project Context

**Repository**: ZKP Dataset Ledger  
**Primary Language**: Rust  
**Domain**: Zero-Knowledge Proofs, Cryptography, ML Pipeline Auditing  
**Architecture**: CLI + Library with modular backends  
**Maturity Level**: Maturing (50-75% SDLC)

## Development Commands

```bash
# Primary development workflow
make dev          # Format, lint, test - run this frequently
make check        # Full quality checks before commits
make pre-release  # Complete release preparation

# Testing and benchmarking
make test-coverage    # Generate HTML coverage reports
make bench           # Run cryptographic benchmarks
make test-integration # Integration tests only

# Security and auditing
make audit          # Cargo audit for vulnerabilities
make outdated       # Check dependency versions

# Docker development
make docker-build   # Build container image
make db-setup      # PostgreSQL development database
```

## Key Project Information

### Technology Stack
- **Cryptography**: Arkworks ecosystem (ark-groth16, ark-bls12-381)
- **Storage**: RocksDB (default), PostgreSQL (optional)
- **CLI**: Clap v4 with comprehensive subcommands
- **Testing**: Criterion benchmarks, proptest property testing
- **Serialization**: Serde with JSON/bincode support

### Architecture Patterns
- **Modular backends**: Storage abstraction (RocksDB/PostgreSQL)
- **Feature flags**: `rocksdb`, `postgres`, `benchmarks`, `property-testing`
- **Zero-copy where possible**: Using Polars for data processing
- **Async throughout**: Tokio runtime for I/O operations

### Critical Files and Directories
- `src/crypto/`: ZK circuit implementations and proof generation
- `src/ledger.rs`: Core ledger and Merkle tree logic
- `src/bin/cli.rs`: Command-line interface implementation
- `benches/`: Cryptographic performance benchmarks
- `tests/integration_tests.rs`: End-to-end testing
- `.pre-commit-config.yaml`: Quality gates (fmt, clippy, test, audit)

### Security Considerations
- **Never commit**: Private keys, test data with real content
- **Trusted setup**: Groth16 requires ceremony parameters
- **Memory safety**: Leverage Rust's guarantees
- **Input validation**: Comprehensive in CLI and API layers
- **Timing attacks**: Consider constant-time operations in crypto

## Development Guidelines

### Code Style
- Follow `rustfmt.toml` and `clippy.toml` configurations
- Use descriptive variable names for cryptographic operations
- Document complex ZK circuit logic extensively
- Prefer explicit error handling with `thiserror`

### Testing Strategy
- **Unit tests**: Each crypto function and ledger operation
- **Integration tests**: Full CLI workflows
- **Property tests**: Cryptographic properties and invariants
- **Benchmarks**: Performance regression detection

### Performance Requirements
- **Proof generation**: Target <5s for 1M row datasets
- **Verification**: <100ms regardless of dataset size
- **Memory usage**: Streaming for datasets >1GB
- **Proof size**: Keep under 1KB for basic operations

## Common Issues and Solutions

### Build Issues
```bash
# Missing system dependencies
sudo apt install cmake clang pkg-config  # Ubuntu/Debian
brew install cmake llvm pkg-config       # macOS

# Rust version too old
rustup update stable
```

### Cryptographic Development
- Use `ark-std::test_rng()` for deterministic testing
- Validate all circuit constraints in debug builds
- Run benchmarks before/after crypto changes
- Test edge cases: empty datasets, single row, very large

### Database Development
```bash
# PostgreSQL feature development
cargo test --features postgres
make db-setup  # Starts test database
```

## AI Agent Instructions

### When working on this project:
1. **Always run `make dev`** before significant changes
2. **Use `make check`** before suggesting commits
3. **Consider cryptographic implications** of any data structure changes
4. **Test with multiple dataset sizes** when modifying core logic
5. **Benchmark performance** for cryptographic operations
6. **Validate security properties** after changes to proof generation

### Code Review Focus Areas:
- Cryptographic correctness (circuit constraints, proof validity)
- Memory safety in unsafe blocks (if any)
- Error handling completeness
- Performance regression in benchmarks
- API ergonomics for CLI users

### Deployment Considerations:
- Docker image size optimization
- Cross-compilation for multiple targets
- Trusted setup parameter distribution
- Documentation currency

## External Dependencies

### Critical Security Dependencies
- `ark-*` crates: Core cryptographic primitives
- `sha3`, `blake3`: Cryptographic hashing
- `rocksdb`: Persistent storage integrity

### Monitor for Updates
- Arkworks ecosystem: Breaking changes possible
- `tokio`: Async runtime compatibility
- `polars`: Data processing performance
- `clap`: CLI interface changes

## Maintenance Schedule

- **Weekly**: Dependency vulnerability scans (`make audit`)
- **Monthly**: Performance regression testing (`make bench`)
- **Quarterly**: Cryptographic library updates and testing
- **Annually**: Security audit of custom crypto implementations

This configuration enables Claude agents to work effectively with the ZKP Dataset Ledger codebase while maintaining security and performance standards.